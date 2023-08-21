#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     :   2023/8/20 12:16 下午
# @Author   :   yvanyao
# @File     :   model_yvan.py
# @Software :   PyCharm

# 首先，我需要一些路径和解析参数相关的包
import pathlib
import pickle

# 其次，需要一些日常的包用于解析数据
import numpy as np

# 还需要一些附加的库，用于显示训练/解析数据过程
import tqdm

# 再者，需要一些专业的库，用于搭建模型和加载数据
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import layers
import evaluation
import sampler as sampler_module


# ************************** ok，现在我们可以开始搭建自己的模块了 *****************
class YvanModel(nn.Module):
    """
    首先，我们会定义一个模型的类，继承自nn.Module，我们需要实现forward函数
    除此之外，我们还会实现一个__init__函数以及repr函数，后者用户获取embedding
    """
    def __init__(self, graph, type, textsets, hidden_dim, n_layers, *args, **kwargs):
        """
        graph：输入的图，基于这个图我们来构建模型
        type：
        textsets：文字类特征
        hidden_dim:中间隐层的长度
        layers：
        """
        super().__init__(*args, **kwargs)
        # 创建一个数据转换函数
        self.proj = layers.LinearProjector(graph, type, textsets, hidden_dim)
        # 创建一个conv卷积层
        self.sage = layers.SAGENet(hidden_dim, n_layers)
        # 创建一个计算得分的层
        self.score = layers.ItemToItemScorer(graph, type)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

    def forward(self, pos_graph, neg_graph, blocks):
        """
        前传函数
        """
        h_item = self.get_repr(blocks)
        pos_scores = self.score(pos_graph, h_item)
        neg_scores = self.score(neg_graph, h_item)

        return (neg_scores - pos_scores + 1).clamp(min=0)


def train(dataset, args):
    """
    定义一个主训练函数，函数里面会有具体的模型训练
    """

    g = dataset["train-graph"]
    item_texts = dataset["item_texts"]
    user_type = dataset["user-type"]
    item_type = dataset["item-type"]

    g.nodes[user_type].data['id'] = torch.arange(g.num_nodes(user_type))
    g.nodes[item_type].data['id'] = torch.arange(g.num_nodes(item_type))

    textsets = {}
    textlist = []
    batch_first = True

    tokenizer = get_tokenizer(None)

    for i in range(g.num_nodes(item_type)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)

    for key, _ in item_texts.items():
        vocab2 = build_vocab_from_iterator(textlist, specials=['<unk>', '<pad>'])
        textsets[key] = {
            textlist,
            vocab2,
            vocab2.get_stoi()['<pad>'],
            batch_first
        }


    # 采样出来三个batch_size的数据
    batch_sampler = sampler_module.ItemToItemBatchSampler(g, user_type, item_type, args.batch_size)

    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_type,
        item_type,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers)

    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_type, textsets
    )

    dataloader = DataLoader(
        batch_sampler,  # 每次生成一个batch，包含heads,tails和neg_tails
        collate_fn=collator.collate_train,  # 由heads+tails+neg_tails生成pos_graph,neg_graph和blocks
        num_workers=args.num_workers,
    )

    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_type)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers
    )

    dataloader_iter = iter(dataloader)

    model = YvanModel(g, item_type, textsets, hidden_dim=args.hidden_dims, n_layers=args.num_layers)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_iter)

            loss = model(pos_graph, neg_graph, blocks)

            opt.zero_grad()
            loss.back_ward()
            opt.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.num_nodes[item_type]).split(args.batch_size)
            h_item_batches = []
            for blocks in dataloader_test:
                h_item_batches.append(model.get_repr(blocks))

            h_item = torch.cat(h_item_batches, 0)

            print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))
