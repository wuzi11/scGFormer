import numpy as np
import os, gc
import torch
import pickle as pkl
import scanpy as sc 
import pandas as pd
from sklearn import preprocessing
from utils.data_utils import rand_train_test_idx
from sklearn.neighbors import kneighbors_graph
from imblearn.over_sampling import SMOTE
import scipy.sparse as sp
import random

class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, train_prop=0.8):
        ignore_negative = True
        valid_prop = 0.0
        train_idx, _, test_idx = rand_train_test_idx(
            self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
        split_idx = {'train': train_idx, 'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))



def load_dataset(data_dir, dataname, use_HVG=False, use_knn=False,
                 n_neighbors=10, max_neighbors=5, sampling_method='random',
                 query_dataset=None):
    if query_dataset:
        raise NotImplementedError("query_dataset")
    else:
        dataset = load_cell_dataset(data_dir, dataname,
                                    use_knn=use_knn,
                                    use_HVG=use_HVG,
                                    n_neighbors=n_neighbors,
                                    max_neighbors=max_neighbors,
                                    sampling_method=sampling_method)
    return dataset 



def load_cell_dataset(data_dir, dataname,
                      use_knn=False, use_HVG=False,
                      n_neighbors=30, max_neighbors=16,
                      sampling_method='random'):
    
    data_path = os.path.join(data_dir, dataname + '.h5ad')
    dataset = NCDataset('scRNA-seq_' + dataname)

    adata = sc.read_h5ad(data_path)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    import scipy.sparse
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X

    if 'cell_ontology_class' in adata.obs:
        y = adata.obs['cell_ontology_class']
    else:
        y = adata.obs['cell_type']

    # if adata.n_vars <= 3500:
    #     HVG = adata.n_vars
    # else:
    HVG = 3000
    del adata
    gc.collect()

    if use_HVG:
        print("HVG is adopted.")
        genes_idx, cells_idx = filter_data(X, highly_genes=HVG)
        X = X[cells_idx][:, genes_idx]
        y = y[cells_idx]
        print('X shape (with HVG selecting): ', X.shape)
    else:
        print("HVG is not adopted.")
        print('X shape (without HVG selecting): ', X.shape)

    le = preprocessing.LabelEncoder()
    le.fit(y)
    class_names = le.classes_
    y = torch.as_tensor(le.transform(y), dtype=torch.long)

    features = torch.as_tensor(X, dtype=torch.float)      
    labels   = y
    
    if use_knn:
        edge_index = construct_knn_graph(
            features=features,
            dataset_name=dataname,
            n_neighbors=n_neighbors,
            max_neighbors=max_neighbors,
            sampling_method=sampling_method
        )
        print("[INFO] Using KNN graph.")
    else:
        edge_index = None

    num_nodes = features.shape[0]
    dataset.graph = {
        'edge_index': edge_index,
        'edge_feat': None,
        'node_feat': features, 
        'num_nodes': num_nodes
    }
    dataset.label = labels  
    dataset.class_names = class_names
    return dataset


def filter_data(X, highly_genes=4000):
    import scanpy as sc
    X_ceil = np.ceil(X).astype(int)
    adata = sc.AnnData(X_ceil, dtype=np.float32)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=4,
        flavor='cell_ranger', min_disp=0.5,
        n_top_genes=highly_genes, subset=True
    )
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)
    return genes_idx, cells_idx


def construct_knn_graph(features, dataset_name,
                        n_neighbors=10, max_neighbors=16,
                        sampling_method='random'):
    import numpy as np
    graph_dir = 'cache/graph'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    knn_graph_path = f'{graph_dir}/{dataset_name}_N{n_neighbors}_maxDeg{max_neighbors}_{sampling_method}.pt'

    if os.path.exists(knn_graph_path):
        print(f'[KNN Graph] Load from {knn_graph_path}')
        edge_index = torch.load(knn_graph_path)
        return edge_index
    else:
        print(f'[KNN Graph] Building knn graph for {dataset_name} with n_neighbors={n_neighbors}, '
              f'max_neighbors={max_neighbors}, method={sampling_method}...')
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        adj = kneighbors_graph(
            X=features,
            n_neighbors=n_neighbors,
            include_self=True,
            mode='distance'
        )
        import scipy.sparse as sp
        if max_neighbors is None or max_neighbors <= 0:
            edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
            torch.save(edge_index, knn_graph_path)
            return edge_index

        adj_coo = adj.tocoo()
        rows = adj_coo.row
        cols = adj_coo.col
        data = adj_coo.data
        row_dict = {}
        for r, c, dist_val in zip(rows, cols, data):
            if r not in row_dict:
                row_dict[r] = []
            row_dict[r].append((c, dist_val))

        new_rows, new_cols, new_data = [], [], []
        for r, neighbors in row_dict.items():
            if len(neighbors) > max_neighbors:
                if sampling_method == 'topk':
                    neighbors.sort(key=lambda x: x[1])
                    neighbors = neighbors[:max_neighbors]
                elif sampling_method == 'random':
                    import numpy as np
                    idxs = np.random.choice(len(neighbors), size=max_neighbors, replace=False)
                    neighbors = [neighbors[i] for i in idxs]
                else:
                    raise ValueError(f"Unknown sampling_method={sampling_method}")

            for c, dist_val in neighbors:
                new_rows.append(r)
                new_cols.append(c)
                new_data.append(dist_val)

        new_adj = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=adj.shape)
        edge_index = torch.tensor(np.vstack(new_adj.nonzero()), dtype=torch.long)

        torch.save(edge_index, knn_graph_path)
        return edge_index