import os

import torch



def load_clearscope_e5():
    graph_5_8 = torch.load("/mnt/hdd.data/hzc/clearscopes-e5/train_graphs/graph_5_8.TemporalData.simple")
    graph_5_9 = torch.load("/mnt/hdd.data/hzc/clearscopes-e5/train_graphs/graph_5_9.TemporalData.simple")
    graph_5_11 = torch.load("/mnt/hdd.data/hzc/clearscopes-e5/train_graphs/graph_5_11.TemporalData.simple")


    return [graph_5_8, graph_5_9, graph_5_11]

def load_clearscope_e3():
    graph_5_8 = torch.load("/mnt/hdd.data/hzc/clearscopes-e3/train_graphs/graph_4_4.TemporalData.simple")
    graph_5_9 = torch.load("/mnt/hdd.data/hzc/clearscopes-e3/train_graphs/graph_4_5.TemporalData.simple")
    graph_5_11 = torch.load("/mnt/hdd.data/hzc/clearscopes-e3/train_graphs/graph_4_6.TemporalData.simple")


    return [graph_5_8, graph_5_9, graph_5_11]

def load_cadets_e3():
    graph_4_2 = torch.load("/mnt/hdd.data/hzc/cadets-e3/train_graphs/graph_4_2.TemporalData.simple")
    graph_4_3 = torch.load("/mnt/hdd.data/hzc/cadets-e3/train_graphs/graph_4_3.TemporalData.simple")
    graph_4_4 = torch.load("/mnt/hdd.data/hzc/cadets-e3/train_graphs/graph_4_4.TemporalData.simple")
    return [graph_4_2, graph_4_3, graph_4_4]

def load_cadets_e5():
    graph_5_8 = torch.load("/mnt/hdd.data/hzc/cadets-e5/train_graphs/graph_5_8.TemporalData.simple")
    graph_5_9 = torch.load("/mnt/hdd.data/hzc/cadets-e5/train_graphs/graph_5_9.TemporalData.simple")
    graph_5_11 = torch.load("/mnt/hdd.data/hzc/cadets-e5/train_graphs/graph_5_11.TemporalData.simple")
    
    return [graph_5_8, graph_5_9, graph_5_11]

def load_theia_e3():
    graph_5_8 = torch.load("/mnt/hdd.data/hzc/theia-e3/train_graphs/graph_4_3.TemporalData.simple")
    graph_5_9 = torch.load("/mnt/hdd.data/hzc/theia-e3/train_graphs/graph_4_4.TemporalData.simple")
    graph_5_11 = torch.load("/mnt/hdd.data/hzc/theia-e3/train_graphs/graph_4_5.TemporalData.simple")
    
    return [graph_5_8, graph_5_9, graph_5_11]

def load_theia_e5():
    graph_5_8 = torch.load("/mnt/hdd.data/hzc/theia-e5/train_graphs/graph_5_8.TemporalData.simple")
    graph_5_9 = torch.load("/mnt/hdd.data/hzc/theia-e5/train_graphs/graph_5_9.TemporalData.simple")
    
    return [graph_5_8, graph_5_9]



