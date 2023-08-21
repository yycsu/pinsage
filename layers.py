import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False

# 这里可以重点研读一下，涉及到多模态接入
def _init_input_modules(g, ntype, textset, hidden_dims):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()   # ModuleDict实例

    for column, data in g.nodes[ntype].data.items():
        # 对id不进行特征处理
        if column == dgl.NID:
            continue
        # 如果特征是float类型，那么创建一个线性层，包含权重的
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        # 如果特征是整形的，那么创建一个Embedding层，且是以data的最大值+2作为长度的matrix
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    # 针对文本类型的特征，我们使用BagOfWords类，自定义一个处理方式
    if textset is not None:
        for column, field in textset.items():
            textlist, vocab, pad_var, batch_first = field
            module_dict[column] = BagOfWords(vocab, hidden_dims)
    # 最后返回不同模式下的Embedding模块
    return module_dict


class BagOfWords(nn.Module):
    def __init__(self, vocab, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(vocab.get_itos()),
            hidden_dims,
            padding_idx=vocab.get_stoi()["<pad>"],
        )
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()


class LinearProjector(nn.Module):
    """
    Projects each input feature of the graph linearly and sums them up
    """

    def __init__(self, full_graph, ntype, textset, hidden_dims):
        super().__init__()

        self.ntype = ntype
        self.inputs = _init_input_modules(
            full_graph, ntype, textset, hidden_dims
        )

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith("__len"):
                # This is an additional feature indicating the length of the ``feature``
                # column; we shouldn't process this.
                continue

            module = self.inputs[feature]
            if isinstance(module, BagOfWords):
                # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + "__len"]
                result = module(data, length)
            else:
                result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            # 将原始特征映射成hidden_dims长，存储在n字段中
            g.srcdata["n"] = self.act(self.Q(self.dropout(h_src)))
            g.edata["w"] = weights.float()
            # src节点的特征n乘以权重，构成m，dst将接收到的消息m，想加起来，存入dst节点的n字段
            # 注意这里的update_all包含的两个函数，第一个是收集message函数，第二个是聚合函数
            # 第一个针对的是src边/节点，生成一个新的m特征，而第二个是将所有m特征聚合生成dst的n特征
            g.update_all(fn.u_mul_e("n", "w", "m"), fn.sum("m", "n"))
            # 这里同样，是先将边的weight特征拷贝成m特征，然后在加和生成dst的ws特征
            g.update_all(fn.copy_e("w", "m"), fn.sum("m", "ws"))
            # 获得dst的加权求和特征
            n = g.dstdata["n"]
            # 获得dst的所有来源边权重和
            ws = g.dstdata["ws"].unsqueeze(1).clamp(min=1)
            # dst的邻居加权求和后与dst节点自身相加，并进行线性变化，激活后得到最终结果
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            # 归一化
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(
                z_norm == 0, torch.tensor(1.0).to(z_norm), z_norm
            )
            z = z / z_norm
            return z


class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        # 逐层卷积，得到最终的embedding
        for _ in range(n_layers):
            self.convs.append(
                WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims)
            )

    def forward(self, blocks, h):
        # 前穿计算过程
        for layer, block in zip(self.convs, blocks):
            h_dst = h[: block.num_nodes("DST/" + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata["weights"])
        return h


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()

        n_nodes = full_graph.num_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes, 1))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {"s": edges.data["s"] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata["h"] = h
            # 边两端节点做点积
            item_item_graph.apply_edges(fn.u_dot_v("h", "h", "s"))
            # 加上首尾两姐弟拿的bias
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata["s"]
        return pair_score
