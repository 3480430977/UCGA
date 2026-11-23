from torch_geometric.nn import conv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from torch import nn, Tensor


class GATConv(conv.GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0,
                 add_self_loop=True, edge_dim=None, fill_value='mean', bias=True, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loop, edge_dim,
                         fill_value, bias, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.leakey_relu = nn.LeakyReLU(negative_slope)

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i, a_j=None, a_i=None):
        a = None if a_j is None or a_i is None else(a_j*a_i).sum(1, True)
        alpha = alpha_j if alpha_i is None else alpha_j+alpha_i
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            alpha += (self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)*self.att_edge).sum(-1)
        if isinstance(a, Tensor):
            alpha = softmax(a*self.leakey_relu(alpha), index, ptr, size_i)
        else:
            alpha = softmax(self.leakey_relu(alpha), index, ptr, size_i)
        return self.dropout(alpha)

    def forward(self, x, edge_index, a=None, edge_attr=None, size=None, return_attention_weights=None):
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, self.heads, self.out_channels)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)
        x = x_src, x_dst
        alpha = (x_src*self.att_src).sum(-1), None if x_dst is None else(x_dst*self.att_dst).sum(-1)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.shape[0])
                edge_index, edge_attr = add_self_loops(*remove_self_loops(edge_index, edge_attr), self.fill_value,
                                                       min(size)if size is not None else num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError("The usage of 'edge_attr' and 'add_self_loops' simultaneously is "
                                              "currently not yet supported for 'edge_index' in a 'SparseTensor' form")
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, a=a)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        out = out.view(-1, self.heads*self.out_channels)if self.concat else out.mean(1)
        if self.bias is not None:
            out += self.bias
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message_and_aggregate(self, adj_t):
        pass


class Classifier(nn.Module):
    def __init__(self, in_channels, n_class, heads, attn_drop, feat_drop):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.gat_conv = GATConv(in_channels, n_class, heads, False, dropout=attn_drop)

    def forward(self, x, edge_index, a):
        return self.gat_conv(self.feat_drop(x), edge_index, a)


class GAT(nn.Module):
    def __init__(self, in_channels, n_hidden, heads, attn_drop, activation, feat_drop, n_layers):
        super().__init__()
        self.activation = activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.gat_layers = nn.ModuleList((GATConv(in_channels, n_hidden, heads, dropout=attn_drop),))
        self.n_layers = n_layers
        for _ in range(1, n_layers):
            self.gat_layers.append(GATConv(n_hidden*heads, n_hidden, heads, dropout=attn_drop))

    def forward(self, x, edge_index, a):
        for _ in range(self.n_layers):
            x = self.activation(self.gat_layers[_](self.feat_drop(x), edge_index, a))
        return x


class Network(nn.Module):
    def __init__(self, in_channels, n_class, n_hidden, heads, out_heads, attn_drop, activation, feat_drop,
                 n_encoder_layers):
        super().__init__()
        self.encoder = GAT(in_channels, n_hidden, heads, attn_drop, activation, feat_drop, n_encoder_layers)
        self.auxiliary_classifier = Classifier(n_hidden*heads, n_class, out_heads, attn_drop, feat_drop)
        self.classifier = Classifier(n_hidden*heads, n_class, out_heads, attn_drop, feat_drop)

    def forward(self, x, edge_index, a):
        x = self.encoder(x, edge_index, a)
        return self.auxiliary_classifier(x, edge_index, a), self.classifier(x, edge_index, a), x
