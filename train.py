from collections import defaultdict
from models import Network
from sklearn.cluster import KMeans
from torch import nn
from torch.optim import Adam
from torch_geometric import seed_everything
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_scatter import scatter_log_softmax, scatter_mean
import numpy as np
import torch
import utils


def _test0(dataset, is_target):
    network.eval()
    with torch.no_grad():
        aux_logit, logit, emb = network(dataset.x, dataset.edge_index, dataset.pred)
    kp1prob = logit.softmax(1)
    loss_kp1 = torch.tensor(dataset.num_nodes*[0], dtype=torch.float, device=dataset.x.device)
    if is_target:
        loss_kp1 += cross_entropy_loss_func_rn(aux_logit, cluster_t.data.positive_unk)
    return emb, kp1prob, loss_kp1.reshape(-1, 1), utils.evaluate(is_target, dataset.y.cpu(), logit.argmax(1).cpu(
    ), kp1prob[:, -1].cpu())if gpu_available else utils.evaluate(is_target, dataset.y, logit.argmax(1), kp1prob[
                                                                                                        :, -1])


def _test1(dataset, is_target):
    network.eval()
    with torch.no_grad():
        _, logit, emb = network(dataset.x, dataset.edge_index, dataset.pred)
    kp1prob = logit.softmax(1)
    return emb, kp1prob, utils.evaluate(is_target, dataset.y.cpu(), logit.argmax(1).cpu(), kp1prob[:, -1].cpu(
    ))if gpu_available else utils.evaluate(is_target, dataset.y, logit.argmax(1), kp1prob[:, -1])


def train0():
    for batch_s, batch_t in zip(loader_s, loader_t):
        # noinspection DuplicatedCode
        indicator_s[:] = False
        indicator_t[:] = False
        indicator_s[batch_s.y.unique()] = True
        batch_t_positive_pseudo_label_unique = batch_t.positive_pseudo_label.unique()
        if batch_t_positive_pseudo_label_unique.shape[0] and batch_t_positive_pseudo_label_unique[0] == -1:
            batch_t_positive_pseudo_label_unique = batch_t_positive_pseudo_label_unique[1:]
        indicator_t[batch_t_positive_pseudo_label_unique] = True
        indicator_s_nonzero = torch.nonzero(indicator_s).reshape(-1)
        indicator_t_nonzero = torch.nonzero(indicator_t).reshape(-1)
        pn = torch.cartesian_prod(indicator_s_nonzero, indicator_t_nonzero)
        mp = pn[:, 0].eq(pn[:, 1])

        mpl = batch_t.positive_pseudo_label > -1
        mtpl = batch_t.top_pseudo_label > -1
        mu = torch.rand(batch_t.num_nodes, dtype=torch.float, device=batch_s.x.device).lt(batch_t.q)
        aux_logit_s, logit_s, emb_s = network(batch_s.x, batch_s.edge_index, batch_s.pred)
        aux_logit_t, logit_t, emb_t = network(batch_t.x, batch_t.edge_index, batch_t.pred)
        loss_s = cross_entropy_loss_func(aux_logit_s, batch_s.y)+cross_entropy_loss_func(logit_s, batch_s.y)
        loss_cont = loss_t = loss_u = 0
        # noinspection DuplicatedCode
        if mp.any():
            centroid_s = scatter_mean(emb_s, batch_s.y, 0, dim_size=batch_s.n_class)
            centroid_t = scatter_mean(emb_t[mpl], batch_t.positive_pseudo_label[mpl], 0, dim_size=batch_t.n_class)
            dpn = cosine_similarity_func_d1(centroid_s[pn[:, 0]], centroid_t[pn[:, 1]])/tau[pn[:, 1]]
            loss_cont += -(scatter_log_softmax(dpn, pn[:, 0])[mp]/batch_s.n_class).sum()
            loss_cont += -(scatter_log_softmax(dpn, pn[:, 1])[mp]/batch_s.n_class).sum()
        if mtpl.any():
            loss_t += cross_entropy_loss_func(logit_t[mtpl], batch_t.top_pseudo_label[mtpl])
        if mu.any():
            loss_u += cross_entropy_loss_func(aux_logit_t[mu], batch_t.positive_unk[mu])
        loss = loss_s+w_cont*loss_cont+w_t*loss_t+w_u*loss_u
        optimizer.zero_grad(True)
        loss.backward()
        optimizer.step()


def train1():
    for batch_s, batch_t in zip(loader_s, loader_t):
        # noinspection DuplicatedCode
        indicator_s[:] = False
        indicator_t[:] = False
        indicator_s[batch_s.y.unique()] = True
        batch_t_positive_pseudo_label_unique = batch_t.positive_pseudo_label.unique()
        if batch_t_positive_pseudo_label_unique.shape[0] and batch_t_positive_pseudo_label_unique[0] == -1:
            batch_t_positive_pseudo_label_unique = batch_t_positive_pseudo_label_unique[1:]
        indicator_t[batch_t_positive_pseudo_label_unique] = True
        indicator_s_nonzero = torch.nonzero(indicator_s).reshape(-1)
        indicator_t_nonzero = torch.nonzero(indicator_t).reshape(-1)
        pn = torch.cartesian_prod(indicator_s_nonzero, indicator_t_nonzero)
        mp = pn[:, 0].eq(pn[:, 1])
        mpl = batch_t.positive_pseudo_label > -1
        mtpl = batch_t.top_pseudo_label > -1
        _, logit_s, emb_s = network(batch_s.x, batch_s.edge_index, batch_s.pred)
        _, logit_t, emb_t = network(batch_t.x, batch_t.edge_index, batch_t.pred)
        loss_s = cross_entropy_loss_func(logit_s, batch_s.y)
        loss_cont = loss_t = 0
        # noinspection DuplicatedCode
        if mp.any():
            centroid_s = scatter_mean(emb_s, batch_s.y, 0, dim_size=batch_s.n_class)
            centroid_t = scatter_mean(emb_t[mpl], batch_t.positive_pseudo_label[mpl], 0, dim_size=batch_t.n_class)
            dpn = cosine_similarity_func_d1(centroid_s[pn[:, 0]], centroid_t[pn[:, 1]])/tau[pn[:, 1]]
            loss_cont += -(scatter_log_softmax(dpn, pn[:, 0])[mp]/batch_s.n_class).sum()
            loss_cont += -(scatter_log_softmax(dpn, pn[:, 1])[mp]/batch_s.n_class).sum()
        if mtpl.any():
            loss_t += cross_entropy_loss_func(logit_t[mtpl], batch_t.top_pseudo_label[mtpl])
        loss = loss_s+w_cont*loss_cont+w_t*loss_t
        optimizer.zero_grad(True)
        loss.backward()
        optimizer.step()


N_UNK = 4
batch_cluster = 64
batch_size = 2048
early_epochs = 10
epochs = 100
lr = 1e-3
weight_decay = 3e-5
attn_drop = 0.6
feat_drop = 0.5
heads = 8
n_encoder_layers = 1
n_hidden = 32
out_heads = 1
a = 0.3
tau_k = 1
tau_u = 0.1
top_k = 0.6
w_cont = 1
w_t = 0.9
w_u = 1
gpu_available = torch.cuda.is_available()
cosine_similarity_func_d1, cross_entropy_loss_func = nn.CosineSimilarity(1), nn.CrossEntropyLoss()
cross_entropy_loss_func_rn = nn.CrossEntropyLoss(reduction='none')
# gpu_available = False
if gpu_available:
    cosine_similarity_func_d1.cuda()
    cross_entropy_loss_func.cuda()
    cross_entropy_loss_func_rn.cuda()
s, t = 'citation-v1', 'dblp-v4'
cluster_size, epochs_p1 = batch_size//batch_cluster, epochs+1
source, target = utils.load_data(s, t, N_UNK)
if gpu_available:
    source.cuda()
    target.cuda()
indicator_s = torch.tensor(source.n_class*[False], device=source.x.device)
indicator_t = torch.tensor(target.n_class*[False], device=source.x.device)
n_cluster_s = source.num_nodes//cluster_size+bool(source.num_nodes % cluster_size)
n_cluster_t = target.num_nodes//cluster_size+bool(target.num_nodes % cluster_size)
indices_s, indices_t = list(range(n_cluster_s)), list(range(n_cluster_t))
max_n_cluster = max(n_cluster_s, n_cluster_t)
aligned_n_cluster_s = max_n_cluster//n_cluster_s+bool(max_n_cluster % n_cluster_s)
aligned_n_cluster_t = max_n_cluster//n_cluster_t+bool(max_n_cluster % n_cluster_t)
source.pred = torch.eye(source.n_class, dtype=torch.int8, device=source.x.device)[source.y]
source.pred[:] = 0
source.pred[:, 0] = 1
target.pred = torch.zeros(target.num_nodes, target.n_class, dtype=torch.float, device=source.x.device)
target.pred[:, 0] = 1
# noinspection PyTypeChecker
tau = torch.tensor(source.n_class*[tau_k]+[tau_u], dtype=torch.float, device=source.x.device)
cluster_s, cluster_t = ClusterData(source, n_cluster_s), ClusterData(target, n_cluster_t)
positive_pseudo_label = torch.tensor(target.num_nodes*[-1], dtype=torch.long, device=source.x.device)
cluster_t.data.positive_pseudo_label = positive_pseudo_label.clone()
cluster_t.data.positive_unk = torch.tensor(target.num_nodes*[source.n_class], dtype=torch.long,
                                           device=source.x.device)
q = torch.tensor(target.num_nodes*[0], dtype=torch.float, device=source.x.device)
cluster_t.data.q = q.clone()
top_pseudo_label = torch.tensor(target.num_nodes*[-1], dtype=torch.long, device=source.x.device)
cluster_t.data.top_pseudo_label = top_pseudo_label.clone()
accs_s, accs_t = {}, {}
os_star_ss, all_ss, os_ts, os_star_ts, all_ts, unk_ts, unk_auc_ts, hs_ts = [], [], [], [], [], [], [], []
for seed in range(5):
    seed_everything(seed)
    activation = nn.ELU(1, True)
    best_accs_s, best_accs_t, best_epoch = defaultdict(float), defaultdict(float), 0
    network = Network(source.num_node_features, target.n_class, n_hidden, heads, out_heads, attn_drop,
                      activation, feat_drop, n_encoder_layers)
    if gpu_available:
        network.cuda()
    optimizer = Adam(network.parameters(), lr, weight_decay=weight_decay)
    indices_s.sort()
    indices_t.sort()
    cluster_t.data.positive_pseudo_label[:] = -1
    cluster_t.data.top_pseudo_label[:] = -1
    cluster_t.data.q[:] = 0
    target.pred[:] = 0
    target.pred[:, 0] = 1
    cluster_t.data.pred[:] = target.pred
    for epoch in range(1, epochs_p1):
        print(f'{s}->{t}, w_u = {w_u}: seed = {seed}, epoch0 = {epoch}')
        np.random.shuffle(indices_s)
        np.random.shuffle(indices_t)
        loader_s = ClusterLoader(cluster_s, batch_size=batch_cluster, sampler=(aligned_n_cluster_s*indices_s)[
                                                                              :max_n_cluster])
        loader_t = ClusterLoader(cluster_t, batch_size=batch_cluster, sampler=(aligned_n_cluster_t*indices_t)[
                                                                              :max_n_cluster])
        if epoch <= early_epochs:
            train0()
            s_emb, *_, accs_s = _test0(source, False)
            t_emb, t_kp1prob, kp1loss, accs_t = _test0(target, True)
            # TODO edge weight
            target.pred[:] = t_kp1prob
            cluster_t.data.pred[:] = target.pred[cluster_t.perm]
            # TODO pseudo-label
            positive_pseudo_label[:] = -1
            cluster_t.data.positive_pseudo_label[:] = -1
            positive_pseudo_label[:] = t_kp1prob.argmax(1)
            cluster_t.data.positive_pseudo_label[:] = positive_pseudo_label[cluster_t.perm]
            # TODO top pseudo-label
            top_pseudo_label[:] = -1
            cluster_t.data.top_pseudo_label[:] = -1
            kmeans = KMeans(2, n_init='auto')
            pseudo_unk = torch.tensor(kmeans.fit_predict(kp1loss.cpu()if gpu_available else kp1loss),
                                      dtype=torch.bool, device=source.x.device)
            cluster_centers = kmeans.cluster_centers_.reshape(-1)
            if cluster_centers[0] < cluster_centers[1]:
                pseudo_unk = ~pseudo_unk
                cluster_centers = cluster_centers[::-1]
            pseudo_unk = pseudo_unk.long()*source.n_class
            mask_k = pseudo_unk == 0
            pseudo_unk[mask_k] = t_kp1prob[mask_k, :-1].argmax(1)
            kp1loss = kp1loss.reshape(-1)
            ikn = pseudo_unk.lt(source.n_class).nonzero(as_tuple=False).reshape(-1)
            top_ikn = ikn[(kp1loss[ikn]-cluster_centers[0]).abs_().argsort()[:int(top_k*ikn.shape[0])]]
            iuk = pseudo_unk.eq(source.n_class).nonzero(as_tuple=False).reshape(-1)
            top_iuk = iuk[(kp1loss[iuk]-cluster_centers[1]).abs_().argsort()[:int(top_k*iuk.shape[0])]]
            top_i = torch.cat((top_ikn, top_iuk))
            top_pseudo_label[top_i] = pseudo_unk[top_i]
            cluster_t.data.top_pseudo_label[:] = top_pseudo_label[cluster_t.perm]
            # TODO q
            q[:] = (t_kp1prob.argmax(1) == source.n_class)*(1-2*a)+a
            cluster_t.data.q[:] = q[cluster_t.perm]
        else:
            train1()
            s_emb, _, accs_s = _test1(source, False)
            t_emb, t_kp1prob, accs_t = _test1(target, True)
            # TODO edge weight
            target.pred[:] = t_kp1prob
            cluster_t.data.pred[:] = target.pred[cluster_t.perm]
            # TODO pseudo-label
            positive_pseudo_label[:] = -1
            cluster_t.data.positive_pseudo_label[:] = -1
            prob_t, pred_t = t_kp1prob.max(1)
            positive_pseudo_label[:] = pred_t
            cluster_t.data.positive_pseudo_label[:] = positive_pseudo_label[cluster_t.perm]
            # TODO top pseudo-label
            top_pseudo_label[:] = -1
            cluster_t.data.top_pseudo_label[:] = -1
            for _ in range(source.n_class):
                i = positive_pseudo_label.eq(_).nonzero(as_tuple=False).reshape(-1)
                top_pseudo_label[i[prob_t[i].topk(int(top_k*i.shape[0]), sorted=False)[1]]] = _
            iuk = positive_pseudo_label.eq(source.n_class).nonzero(as_tuple=False).reshape(-1)
            top_pseudo_label[iuk[prob_t[iuk].topk(int(top_k*iuk.shape[0]), sorted=False)[1]]] = source.n_class
            cluster_t.data.top_pseudo_label[:] = top_pseudo_label[cluster_t.perm]
        if accs_t['OS']+accs_t['OS*']+accs_t['UNK_AUC']+accs_t['HS'] >= best_accs_t['OS']+best_accs_t['OS*'] +\
                best_accs_t['UNK_AUC']+best_accs_t['HS']:
            best_accs_s, best_accs_t, best_epoch = accs_s, accs_t, epoch
        print('source: (OS*: {:.4f}, ALL: {:.4f})'.format(accs_s['OS*'], accs_s['ALL']))
        print('target: (OS: {:.4f}, OS*: {:.4f}, ALL: {:.4f}, UNK: {:.4f}, UNK_AUC: {:.4f}, HS: {:.4f})'
              ''.format(accs_t['OS'], accs_t['OS*'], accs_t['ALL'], accs_t['UNK'], accs_t['UNK_AUC'],
                        accs_t['HS']))
    os_star_ss.append(accs_s['OS*'])
    all_ss.append(accs_s['ALL'])
    os_ts.append(accs_t['OS'])
    os_star_ts.append(accs_t['OS*'])
    all_ts.append(accs_t['ALL'])
    unk_ts.append(accs_t['UNK'])
    unk_auc_ts.append(accs_t['UNK_AUC'])
    hs_ts.append(accs_t['HS'])
print('mean source: (OS*: {:.4f}±{:.4f}, ALL: {:.4f}±{:.4f})'.format(np.mean(os_star_ss), np.std(os_star_ss),
                                                                     np.mean(all_ss), np.std(all_ss)))
print('mean target: (OS: {:.4f}±{:.4f}, OS*: {:.4f}±{:.4f}, ALL: {:.4f}±{:.4f}, UNK: {:.4f}±{:.4f}, '
      'UNK_AUC: {:.4f}±{:.4f}, HS: {:.4f}±{:.4f})'.format(np.mean(os_ts), np.std(os_ts), np.mean(os_star_ts),
                                                          np.std(os_star_ts), np.mean(all_ts), np.std(all_ts),
                                                          np.mean(unk_ts), np.std(unk_ts),
                                                          np.mean(unk_auc_ts), np.std(unk_auc_ts),
                                                          np.mean(hs_ts), np.std(hs_ts)))
