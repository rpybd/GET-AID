from collections import defaultdict

import torch

from config import *


class NewMemory:
    def __init__(self, num_nodes: int, memory_dim: int):
        """
        初始化Memory类，设置节点数量、内存维度以及时间戳维度。

        :param num_nodes: 节点的数量
        :param memory_dim: 每个节点内存的维度
        :param time_dim: 时间戳的维度
        """
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        
        # 字典用于存储每个节点的内存
        self.memory_dict = torch.randn(num_nodes, memory_dim, device=device)
        self.time_dict = torch.zeros(num_nodes, dtype=torch.long, device=device)
        # self.memory_dict = {}
        
        # # 字典用于存储每个节点的时间戳
        # self.time_dict = {}

        # for i in range(num_nodes):
        #     self.memory_dict[i] = torch.randn(self.memory_dim)
        #     self.time_dict[i] = 0

    def update_memory(self, n_ids: torch.tensor, new_embedding: torch.tensor, ema_alpha: float = 0.95):
        """
        使用指数加权平均（EMA）算法更新指定节点的内存。

        :param n_id: 节点ID
        :param new_embedding: 新的嵌入向量，大小为 (memory_dim,)
        :param ema_alpha: EMA的衰减系数，默认为0.95
        """
        # 使用 defaultdict 来存储每个n_id对应的所有嵌入向量
        assert torch.max(n_ids).item() <= self.num_nodes
        embedding_accumulator = defaultdict(list)
        
        # 将相同 n_id 的 embedding 进行分组
        for i, n_id in enumerate(n_ids):
            n_id = n_id.item()  # 转换为Python整数
            embedding_accumulator[n_id].append(new_embedding[i])
            
        # 对每个n_id，计算平均嵌入向量，然后进行内存更新
        for n_id, embeddings_list in embedding_accumulator.items():
            # 计算平均的embedding
            avg_embedding = torch.mean(torch.stack(embeddings_list), dim=0)
            current_memory = self.memory_dict[n_id]
            # 使用EMA算法更新内存
            updated_memory = ema_alpha * current_memory + (1 - ema_alpha) * avg_embedding
            self.memory_dict[n_id] = updated_memory


    def update_time(self, n_ids: torch.tensor, new_time: torch.tensor):
        """
        更新节点n_id的时间戳。如果节点n_id已经存在，取已有时间戳的中位数。

        :param n_id: 节点ID
        :param new_time: 新的时间戳
        """

        assert torch.max(n_ids).item() <= self.num_nodes
        time_accumulator = defaultdict(list)
        
        # 将相同 n_id 的时间戳进行分组
        for i, n_id in enumerate(n_ids):
            n_id = n_id.item()  # 转换为Python整数
            time_accumulator[n_id].append(new_time[i].item())
        
        # 对每个n_id，计算中位数的时间戳，然后进行更新
        for n_id, times_list in time_accumulator.items():
            # 计算时间戳的中位数
            median_time = torch.median(torch.tensor(times_list, dtype=torch.long))
            self.time_dict[n_id] = median_time
    
    def update(self, src, dst, src_embedding, dst_embedding, t):
        self.update_memory(src, src_embedding)
        self.update_memory(dst, dst_embedding)
        self.update_time(src, t)
        self.update_time(dst, t)
    
    def update2(self, node, embedding, src, dst, t):
        self.update_memory(node, embedding)
        self.update_time(src, t)
        self.update_time(dst, t)

    def __call__(self, n_ids: torch.tensor):
        """
        提取指定节点n_id的内存和时间戳。

        :param n_id: 节点ID
        :return: (memory, time) 其中memory为节点的内存，time为节点的时间戳
        """

        assert torch.max(n_ids).item() <= self.num_nodes
        memories = list()
        times = list()

        for n_id in n_ids:
            n_id = n_id.item()  # 转换为Python整数
            memory = self.memory_dict[n_id]
            time = self.time_dict[n_id]
            memories.append(memory)
            times.append(time)
        return torch.stack(memories), torch.tensor(times)

if __name__=='__main__':
    memory = NewMemory(
        num_nodes=15,
        memory_dim=100,
    )
    n_id = torch.tensor([2,5,6,2,2])
    message = torch.randn(5, 100)    
    times = torch.tensor([1522988100361000000, 1522989024614000000, 1522988702903000000,1522989025899000000,1522989025895000000])
    
    for i in range(5):
        memory.update(n_id, message)
        
        z, last_update = memory(n_id)
        print(z)
        print(z.shape)
        print(last_update)
        memory.update_time(n_id, times)
        print('========')
        