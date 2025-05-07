# encoding=utf-8
import copy
import datetime
import gc
import json
import math
import os
import os.path as osp
import re
import time
from datetime import datetime, timezone
from random import choice
from time import mktime
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2

# import pytorch_lightning as pl
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
import xxhash
from accelerate import Accelerator
from psycopg2 import extras as ex
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric import *
from torch_geometric.data import Data, TemporalData
from torch_geometric.datasets import ICEWS18, JODIEDataset
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    MeanAggregator,
)
from torch_geometric.utils import negative_sampling, to_networkx
from tqdm import tqdm

# accelerator = Accelerator()

# device = accelerator.device
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['RANK']='0'


# We conducted this experiment on CPU, including training, loading and testing models.
# For reproducibility, we recommend users to load and test our models on CPU.

# gpus = [0, 1, 2, 3]
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))

# msg structure:    [src_node_feature,edge_attr,dst_node_feature]

# compute the best partition


# import community as community_louvain

# Find the edge index which the edge vector is corresponding to
def cal_pos_edges_loss(link_pred_ratio):
    loss=[]
    for i in link_pred_ratio:
        loss.append(criterion(i,torch.ones(1)))
    return torch.tensor(loss)

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels,criterion):
    loss=[] 
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)

def cal_pos_edges_loss_autoencoder(decoded,msg):
    loss=[] 
    for i in range(len(decoded)):
        loss.append(criterion(decoded[i].reshape(1,-1),msg[i].reshape(-1)))
    return torch.tensor(loss)

def tensor_find(t, x):
    t_np = t.cpu().numpy()
    idx = np.argwhere(t_np == x)
    return idx[0][0] + 1


def std(t):
    t = np.array(t)
    return np.std(t)


def var(t):
    t = np.array(t)
    return np.var(t)


def mean(t):
    t = np.array(t)
    return np.mean(t)


def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()

def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    s += "." + str(int(int(ns) % 1000000000)).zfill(9)
    return s


def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone("US/Eastern")
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    s += "." + str(int(int(ns) % 1000000000)).zfill(9)
    return s


def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone("US/Eastern")
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")

    return s


def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp


def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone("US/Eastern")
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)


def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone("US/Eastern")
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)

class TemporalDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges succesive events of a
    :class:`torch_geometric.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
            from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            destination nodes to the number of postive destination nodes.
            (default: :obj:`0.0`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        data: TemporalData,
        batch_size: int = 1,
        neg_sampling_ratio: float = 0.0,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', None)

        self.data = data
        self.events_per_batch = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio

        if neg_sampling_ratio > 0:
            self.min_dst = int(data.dst.min())
            self.max_dst = int(data.dst.max())

        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = range(0, len(data) - batch_size, batch_size)
        else:
            arange = range(0, len(data), batch_size)

        super().__init__(arange, 1, shuffle=False, collate_fn=self, **kwargs)

    def __call__(self, arange: List[int]) -> TemporalData:
        batch = self.data[arange[0]:arange[0] + self.events_per_batch]

        n_ids = [batch.src, batch.dst]

        if self.neg_sampling_ratio > 0:
            batch.neg_dst = torch.randint(
                low=self.min_dst,
                high=self.max_dst + 1,
                size=(round(self.neg_sampling_ratio * batch.dst.size(0)), ),
                dtype=batch.dst.dtype,
                device=batch.dst.device,
            )
            n_ids += [batch.neg_dst]

        batch.n_id = torch.cat(n_ids, dim=0).unique()

        return batch

class getComponent:
    def __init__(self, n_id, edge_index):
        self.data = Data(x=n_id, edge_index=edge_index)
        # print("节点 ", data.x)
        # print(f"src{src}, dst{pos_dst}")
        # print("原始边 ", data.edge_index)
        undirected_edge_index = self.directed_to_undirected(self.data.edge_index)
        self.data.edge_index = undirected_edge_index


    # 2. 定义DFS来查找连通域
    def dfs(self, graph, node, visited, component):
        visited[node] = True
        component.append(node)
        neighbors = list(graph.neighbors(node))

        for neighbor in neighbors:
            if not visited[neighbor]:
                self.dfs(graph, neighbor, visited, component)

    def find_connected_components(self):
        graph = to_networkx(self.data, to_undirected=True)
        visited = [False] * graph.number_of_nodes()
        connected_components = []

        for node in range(graph.number_of_nodes()):
            if not visited[node]:
                component = []
                self.dfs(graph, node, visited, component)
                connected_components.append(component)

        return connected_components
    
    def  directed_to_undirected(self, edge_index):
        row, col = edge_index
        # 将边反向
        reversed_edges = torch.stack([col, row], dim=0)
        # 合并原有边和反向边
        undirected_edge_index = torch.cat([edge_index, reversed_edges], dim=1)
        return undirected_edge_index