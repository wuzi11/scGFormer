import argparse
import logging
import os
import random
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from utils.dataset import load_dataset
from utils.model import scGFformer, info_nce_loss
from utils.loss import BalancedSoftmaxCE, FocalLoss, LDAMLoss

warnings.filterwarnings("ignore")


def get_logger(save_dir: str = "./logs", filename: Optional[str] = None) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or "train.log"
    log_path = os.path.join(save_dir, filename)

    logger = logging.getLogger("train_graphformer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def ms(values: List[float]) -> str:
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def adaptive_sparse_perturbation(
    x: torch.Tensor,
    cell_type_labels: Optional[torch.Tensor] = None,
    noise_scale: float = 0.05,
    zero_activate_prob: float = 0.02,
) -> torch.Tensor:
    device = x.device
    perturbed_x = x.clone()
    batch_size, _ = x.shape
    zero_mask = (x == 0)
    non_zero_mask = ~zero_mask

    if cell_type_labels is not None and len(cell_type_labels) > 0:
        unique_types, type_counts = torch.unique(cell_type_labels, return_counts=True)
        type_freq = type_counts.float() / len(cell_type_labels)

        adaptive_noise_scale = torch.zeros(batch_size, device=device)
        adaptive_zero_prob = torch.zeros(batch_size, device=device)

        for i, cell_type in enumerate(cell_type_labels):
            type_idx = (unique_types == cell_type).nonzero(as_tuple=True)[0]
            freq = type_freq[type_idx].item()
            if freq < 0.05:
                adaptive_noise_scale[i] = noise_scale * 0.5
                adaptive_zero_prob[i] = zero_activate_prob * 0.5
            else:
                adaptive_noise_scale[i] = noise_scale
                adaptive_zero_prob[i] = zero_activate_prob
    else:
        adaptive_noise_scale = torch.full((batch_size,), noise_scale, device=device)
        adaptive_zero_prob = torch.full((batch_size,), zero_activate_prob, device=device)

    for i in range(batch_size):
        cell_non_zero = non_zero_mask[i]
        if cell_non_zero.any():
            cell_noise_scale = adaptive_noise_scale[i]
            relative_noise = torch.randn(cell_non_zero.sum(), device=device) * cell_noise_scale
            perturbed_x[i, cell_non_zero] *= (1 + relative_noise)

    perturbed_x = torch.clamp(perturbed_x, min=0)
    return perturbed_x


def vanilla_random_mask(x: torch.Tensor, mask_prob: float) -> torch.Tensor:
    if mask_prob <= 0:
        return x
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


def main(args: argparse.Namespace) -> None:
    logger = get_logger(save_dir="./logs", filename=f"{args.dataset}_contrast.log")
    logger.info(
        "PARAM | dataset=%s use_HVG=%s use_knn=%s hidden=%d heads=%d loss=%s lr=%.1e wd=%.1e "
        "epochs=%d seed=%d contrastive=%s lambda_cl=%.2f temp=%.2f dropgene=%.2f patience=%d",
        args.dataset, args.use_HVG, args.use_knn, args.hidden_channels, args.num_heads, args.loss,
        args.lr, args.weight_decay, args.epochs, args.seed, args.contrastive, args.lambda_cl,
        args.temperature, args.dropgene_prob, args.patience,
    )

    fix_seed(args.seed)
    device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available() and not args.cpu
        else torch.device("cpu")
    )

    dataset = load_dataset(
        data_dir=args.data_dir,
        dataname=args.dataset,
        use_HVG=args.use_HVG,
        use_knn=args.use_knn,
    )
    graph = dataset.graph
    node_feat: torch.Tensor = graph["node_feat"].to(device)
    edge_index = graph["edge_index"]
    if edge_index is not None:
        edge_index = edge_index.to(device)
    label: torch.Tensor = dataset.label.to(device)

    num_nodes, num_features = node_feat.shape
    num_classes = int(label.max()) + 1
    all_idx = torch.arange(num_nodes, device=device)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    val_accs, val_pres, val_recs, val_f1s = [], [], [], []
    f1_record = []

    for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(all_idx.cpu().numpy()), 1):
        train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=device)
        train_loader = DataLoader(TensorDataset(train_idx), batch_size=4096, shuffle=True)

        train_label_counts = torch.bincount(label[train_idx], minlength=num_classes).float().to(device)
        beta = 0.9999
        effective_num = 1.0 - beta ** train_label_counts
        class_weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
        class_weights = (class_weights / class_weights.sum()) * num_classes
        class_weights = class_weights.to(device)
        log_prior = torch.log(torch.clamp(train_label_counts / train_label_counts.sum(), min=1e-12))

        model = scGFformer(
            in_dim=num_features,
            hidden_dim=args.hidden_channels,
            out_dim=num_classes,
            heads=args.num_heads,
            use_performer=args.use_performer,
            performer_dim=args.performer_dim,
            use_gat=args.use_gat,
            use_pe=args.use_pe,
            use_gene_att=args.use_gene_att,
            dynamic_graph=args.dynamic_graph,
            contrastive=args.contrastive,
        ).to(device)
        model.apply(reset_parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.loss == "ce":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "weighted_ce":
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif args.loss == "focal":
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        elif args.loss == "balanceCE":
            criterion = BalancedSoftmaxCE(0.25 * log_prior)
        else:
            criterion = LDAMLoss(train_label_counts, max_m=0.5, s=30, weight=class_weights)

        best_val_acc = best_pre = best_rec = best_f1 = 0.0
        early_stop_counter = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for (batch_indices,) in train_loader:
                idx = batch_indices
                optimizer.zero_grad()

                if args.contrastive:
                    x1 = node_feat

                    if args.vanilla_contrastive:
                        x2 = node_feat.clone()
                        x2[idx] = vanilla_random_mask(node_feat[idx], mask_prob=args.dropgene_prob)
                    else:
                        x2 = node_feat.clone()
                        batch_labels = label[idx]
                        x2[idx] = adaptive_sparse_perturbation(
                            x=node_feat[idx],
                            cell_type_labels=batch_labels,
                            noise_scale=args.dropgene_prob,
                            zero_activate_prob=0.01,
                        )

                    logits1, proj1 = model(x1, edge_index, return_projection=True)
                    logits2, proj2 = model(x2, edge_index, return_projection=True)

                    ce_loss = criterion(logits1[idx], label[idx])
                    cl_loss = info_nce_loss(proj1[idx], proj2[idx], temperature=args.temperature)
                    loss = args.alpha * ce_loss + args.lambda_cl * cl_loss
                else:
                    logits = model(node_feat, edge_index)
                    loss = criterion(logits[idx], label[idx])

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= max(1, len(train_loader))
            
            model.eval()
            with torch.no_grad():
                logits = model(node_feat, edge_index)
                if isinstance(logits, tuple):
                    logits = logits[0]
                val_logits = logits[val_idx]
                val_pred = val_logits.argmax(dim=-1)
                val_acc = (val_pred == label[val_idx]).float().mean().item()

                pred_np = val_pred.detach().cpu().numpy()
                true_np = label[val_idx].detach().cpu().numpy()
                val_pre = precision_score(true_np, pred_np, average="macro", zero_division=0)
                val_rec = recall_score(true_np, pred_np, average="macro", zero_division=0)
                val_f1 = f1_score(true_np, pred_np, average="macro", zero_division=0)

            logger.info(
                "Fold %d | Epoch %02d | loss %.4f | acc %.4f | pre %.4f | rec %.4f | f1 %.4f | no_improve %d/%d",
                fold, epoch, epoch_loss, val_acc, val_pre, val_rec, val_f1, early_stop_counter, args.patience
            )

            f1_record.append({
                "dataset": args.dataset,
                "fold": fold,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_pre": val_pre,
                "val_rec": val_rec,
                "val_f1": val_f1,
            })

            if val_f1 > best_f1:
                best_val_acc, best_pre, best_rec, best_f1 = val_acc, val_pre, val_rec, val_f1
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= args.patience:
                logger.info("Fold %d | Early stopping at epoch %d (no improvement for %d epochs)",
                            fold, epoch, args.patience)
                break

        val_accs.append(best_val_acc)
        val_pres.append(best_pre)
        val_recs.append(best_rec)
        val_f1s.append(best_f1)

        train_label_counts = torch.bincount(label[train_idx], minlength=num_classes).float()
        train_total = train_label_counts.sum().item()
        train_label_ratio = train_label_counts / max(1.0, train_total)
        k_min = min(5, num_classes)
        _, minority_indices = torch.topk(train_label_counts, k=k_min, largest=False)

        with torch.no_grad():
            val_pred_np = val_pred.detach().cpu().numpy()
            val_true_np = label[val_idx].detach().cpu().numpy()
            logger.info("Minority %d classes in train (class_id, train_ratio, val_acc):", k_min)
            for c in minority_indices.detach().cpu().numpy():
                mask = (val_true_np == c)
                n_samples = mask.sum()
                acc_c = float((val_pred_np[mask] == c).sum() / n_samples) if n_samples > 0 else float("nan")
                logger.info("  Class %d: ratio=%.4f, val_acc=%.4f", int(c), train_label_ratio[int(c)].item(), acc_c)

    acc_summary, pre_summary = ms(val_accs), ms(val_pres)
    rec_summary, f1_summary = ms(val_recs), ms(val_f1s)
    print("===== 5-fold summary =====")
    print("ACC |", acc_summary)
    print("PRE |", pre_summary)
    print("REC |", rec_summary)
    print("F1  |", f1_summary)
    logger.info("===== 5-fold summary =====")
    logger.info("ACC | %s", acc_summary)
    logger.info("PRE | %s", pre_summary)
    logger.info("REC | %s", rec_summary)
    logger.info("F1  | %s", f1_summary)

    import pandas as pd
    os.makedirs("./logs", exist_ok=True)
    csv_path = f"./logs/{args.dataset}_f1_per_epoch.csv"
    pd.DataFrame(f1_record).to_csv(csv_path, index=False)
    logger.info("Per-epoch F1 saved to %s", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--use_HVG", action="store_true")
    parser.add_argument("--use_knn", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--use_performer", action="store_true")
    parser.add_argument("--performer_dim", type=int, default=128)
    parser.add_argument("--use_gat", action="store_true")
    parser.add_argument("--use_pe", action="store_true")
    parser.add_argument("--use_gene_att", action="store_true")
    parser.add_argument("--dynamic_graph", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "weighted_ce", "focal", "balanceCE", "LDAMLoss"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lambda_cl", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for classification loss")
    parser.add_argument("--dropgene_prob", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--vanilla_contrastive", action="store_true",
                        help="Use vanilla random masking for contrastive view")
    args = parser.parse_args()
    main(args)