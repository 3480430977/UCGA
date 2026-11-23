from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.data import Data
import numpy as np
import scipy.io as sio
import torch
LABEL_SET = ('人工智能', '人机交互与普适计算', '信息安全', '数据库/数据挖掘/内容检索', '计算机体系结构/并行与分布计算/存储系统',
             '计算机图形学与多媒体', '计算机科学理论', '计算机网络', '软件工程/系统软件/程序设计语言')  # C D
# LABEL_SET = tuple(range(10))  # A1 A2
# LABEL_SET = tuple(range(18))  # L1 L2
# LABEL_SET = tuple(range(40))  # OA1 OA2
N_CLASS = len(LABEL_SET)


def evaluate(is_target, y_true, y_pred, y_score):
    accs, label_set, scores = {}, y_true.unique(), []
    for label in label_set:
        mask = y_true == label
        acc = accuracy_score(y_true[mask], y_pred[mask])
        scores.append(acc)
        if is_target:
            if label < label_set.shape[0]-1:
                accs[LABEL_SET[label]] = acc
        else:
            accs[LABEL_SET[label]] = acc
    if is_target:
        mask = y_true != label_set[-1]
        acc_k = accuracy_score(y_true[mask], y_pred[mask])
        accs.update({'OS': np.mean(scores), 'OS*': np.mean(scores[:-1]), 'ALL': accuracy_score(y_true, y_pred),
                     'UNK': scores[-1], 'UNK_AUC': roc_auc_score(y_true == label_set[-1], y_score),
                     'HS': 2*acc_k*scores[-1]/(acc_k+scores[-1]+1e-9)})
    else:
        accs.update({'OS*': np.mean(scores), 'ALL': accuracy_score(y_true, y_pred)})
    return accs


def load_data(source, target, n_unk):
    source, target, unk_label = to_pyg_data(source), to_pyg_data(target), N_CLASS-n_unk
    source = source.subgraph(source.y < unk_label)
    source.n_class, target.n_class = unk_label, unk_label+1
    target.y[target.y > unk_label] = unk_label
    return source, target


def to_pyg_data(dataset):
    dataset = sio.loadmat(f'data/{dataset}')
    adj = dataset['adjacency_matrix'].tocoo(False)
    return Data(torch.tensor(dataset['features'].toarray(), dtype=torch.float), torch.stack((torch.tensor(
        adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long))), y=torch.tensor(dataset['labels'][0],
                                                                                              dtype=torch.long))
