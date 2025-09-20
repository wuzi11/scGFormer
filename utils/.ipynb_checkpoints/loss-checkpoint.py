import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss
    

class BalancedSoftmaxCE(nn.Module):
    def __init__(self, log_prior):
        super().__init__()
        self.register_buffer("log_prior", log_prior)

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_prior, target)
    
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float)))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = nn.Parameter(m_list, requires_grad=False)
        self.s = s
        self.weight = weight

    def forward(self, logits, targets):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        m_list = self.m_list.to(targets.device)
        batch_m = m_list[targets]
        logits_m = logits.clone()
        logits_m[index] -= batch_m
        return F.cross_entropy(self.s * logits_m, targets, weight=self.weight)
    
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    参考: https://arxiv.org/abs/2004.11362
    - features: [N, D]  已经 L2-normalize
    - labels  : [N]     若为 None → 无监督(同实例正样本)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        """
        返回一个标量 loss
        """
        device = features.device
        if labels is None:
            labels = torch.arange(len(features), device=device)

        # 正样本掩码
        labels = labels.view(-1, 1)                       # [N,1]
        mask = torch.eq(labels, labels.T).float()         # [N,N]
        mask = mask.fill_diagonal_(0)                     # 自身不算正

        # 余弦相似度 / temperature
        sim = features @ features.T                       # [N,N]
        sim = sim / self.T
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)  # 数值稳定
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim)

        # 计算对比损失
        pos_exp = exp_sim * mask
        neg_exp = exp_sim * (1 - mask)
        pos_sum = pos_exp.sum(dim=1)
        denom   = pos_sum + neg_exp.sum(dim=1)
        loss = -torch.log((pos_sum + 1e-8) / (denom + 1e-8))
        return loss.mean()