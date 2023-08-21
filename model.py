import argparse
import os
import pickle

import dgl

import evaluation
import layers
import numpy as np
import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

"""
这个脚本的结构很简单，主要包含一个类PinSAGEModel和一个函数train
1、PinSAGEModel继承了nn.Module类，重新实现了forward函数，当然forward函数中需要使用到repr函数，可以获取最多96个节点的embedding表达
2、每个batch，我们会组成batch_size个正样本和batch_size个负样本，这样就可以去构造hinge loss损失函数
3、DataLoader中的collate_fn函数比较重要，每次从迭代器中获得采样节点，并构造正负样本集
4、最终，将数据传入model，计算loss并且反传回来，更新权重，完成训练
"""

# 下面我们来研究一下PinSAGEModel这个类
class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()
        # 将输入数据转换成embedding
        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims
        )
        # 得到卷积层
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        # 对pos_graph和neg_graph每条边打分，逻辑是使用某边两端节点的点积，再加上两端节点的bias
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        # 得到batch中heads+tails+neg_tails这些节点的embedding
        h_item = self.get_repr(blocks)
        # 得到heads->tails这些边上的得分
        pos_score = self.scorer(pos_graph, h_item)
        # 得到heads->neg_tails这些边上的得分
        neg_score = self.scorer(neg_graph, h_item)
        # neg_score-pos_score再加一，返回一个margin hinge loss，这里margin是1，hinge_loss这里也要重点看一下
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        # 这里的blocks单层里面，src是邻居节点列表，而dst是被采样邻居的节点，所以src的数量显然是大于dst的
        # 输入节点上的原始特征映射成hidden_dims长的向量
        h_item = self.proj(blocks[0].srcdata)
        # 输出节点上的原始特征映射成hidden_dims长的向量
        h_item_dst = self.proj(blocks[-1].dstdata)
        # 多层卷积，得到输出节点的卷积结果，再加上输出节点的原始特征映射结果，得到输出节点上的最终向量表示
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    for i in range(g.num_nodes(item_ntype)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)
    for key, field in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"]
        )
        textset[key] = (
            textlist,
            vocab2,
            vocab2.get_stoi()["<pad>"],
            batch_first,
        )

    # Sampler
    # 采样出三个batch_size大小的节点列表，heads，tails，neg_tails
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size
    )
    # heads->tails生成positive graph
    # heads->neg_tails生成negative graph
    # heads + tails + neg_tails 反向搜索，生成block
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset
    )   # 调用neighbor_sampler 为这个batch的heads，tails，neg_tails；根据heads,tails,neg_tails，生成pos_graph,neg_graph和blocks；然后将原图中节点的特征拷贝进blocks节点
    dataloader = DataLoader(
        batch_sampler,  # 每次生成一个batch，包含heads,tails和neg_tails
        collate_fn=collator.collate_train,  # 由heads+tails+neg_tails生成pos_graph,neg_graph和blocks
        num_workers=args.num_workers,
    )
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        # 由于其中生成的block用到了邻居采样，所以也只能用于训练时的测试，不能用于生成线上的向量
        # 真正线上使用的向量，必须拿一个节点的所有邻居进行卷积得到
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):    # 设定一个epoch中有20000个batch
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                args.batch_size
            )
            h_item_batches = []
            for blocks in dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)

            print(
                evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)
            )


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)       # 数据集的路径
    parser.add_argument("--random-walk-length", type=int, default=2)        # 随机游走最远走的距离
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)      # 从新开始随机游走的概率
    parser.add_argument("--num-random-walks", type=int, default=10)     # 采样时，随机游走的次数
    parser.add_argument("--num-neighbors", type=int, default=3)     # 每个节点找出它随机游走概率最大的3个邻居节点
    parser.add_argument("--num-layers", type=int, default=2)    # 卷积层数量
    parser.add_argument("--hidden-dims", type=int, default=16)  # 隐层维度
    parser.add_argument("--batch-size", type=int, default=32)   # batch_size大小
    parser.add_argument(
        "--device", type=str, default="cpu"
    )  # can also be "cuda:0"
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, "data.pkl")
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset["train-graph"] = g_list[0]
    train(dataset, args)
