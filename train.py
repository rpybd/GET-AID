# encoding=utf-8
import logging

from config import *
from data import *
from model import *

# os.system(f"mkdir -p {artifact_dir}")
# os.system(f"mkdir -p {models_dir}")

logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

connect = psycopg2.connect(
    database=database,
    host="localhost",
    user="postgres",
    password="123456",
    port="5432",
)

cur = connect.cursor()



class getComponent:
    def __init__(self, n_id, edge_index, batch):
        self.data = Data(x=n_id, edge_index=edge_index)
        # print("节点 ", data.x)
        # print(f"src{src}, dst{pos_dst}")
        # print("原始边 ", data.edge_index)
        undirected_edge_index = self.directed_to_undirected(self.data.edge_index)
        self.data.edge_index = undirected_edge_index

        self.batch = batch
        self.base_folder = "./dfs_visual_test"

    # 2. 定义DFS来查找连通域
    def dfs(self, graph, node, visited, component):
        visited[node] = True
        component.append(node)
        neighbors = list(graph.neighbors(node))

        for neighbor in neighbors:
            if not visited[neighbor]:
                self.dfs(graph, neighbor, visited, component)

    def find_connected_components(self, graph):
        visited = [False] * graph.number_of_nodes()
        connected_components = []

        for node in range(graph.number_of_nodes()):
            if not visited[node]:
                component = []
                self.dfs(graph, node, visited, component)
                connected_components.append(component)

        return connected_components

    def directed_to_undirected(self, edge_index):
        row, col = edge_index
        # 将边反向
        reversed_edges = torch.stack([col, row], dim=0)
        # 合并原有边和反向边
        undirected_edge_index = torch.cat([edge_index, reversed_edges], dim=1)
        return undirected_edge_index

    def visualize(self):
        G = to_networkx(self.data, to_undirected=True)
        connected_components = self.find_connected_components(G)
        pos = nx.spring_layout(G)  # 为图的布局生成位置

        for i, component in enumerate(connected_components):
            component_folder = os.path.join(self.base_folder, str(self.batch))
            if not os.path.exists(component_folder):
                os.makedirs(component_folder)
            plt.figure()
            subgraph = G.subgraph(component)
            labels = {node: str(node) for node in subgraph.nodes()}
            nx.draw(
                subgraph,
                pos,
                labels=labels,
                with_labels=True,
                node_color=f"C{i}",
                node_size=700,
                font_size=16,
            )
            plt.title(f"连通域 {i + 1}")
            file_path = os.path.join(component_folder, f"component_{i + 1}.png")
            plt.savefig(file_path)
            plt.close()  # 关闭当前绘制窗口，避免占用内存
            plt.show()
        log_file = os.path.join(component_folder, f"batch{self.batch}.log")
        with open(log_file, "w") as f:
            f.write(f"节点{self.data.x}\n")
            f.write(f"边{self.data.edge_index}\n")
            f.write(f"连通域{connected_components}\n")


# Helper vector to map global node indices to local ones.


def train(
    train_data,
    memory,
    gnn,
    link_pred,
    neighbor_loader,
    optimizer,
    criterion,
    assoc,
    BATCH,
):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes = set()

    total_loss = 0
    for batch in train_data.seq_batches(batch_size=BATCH):
        # batch:
        # TemporalData(dst=[1024], msg=[1024, 42], src=[1024], t=[1024])
        # 按照t的顺序取batch
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        
        # edge_num += edge_index.shape[1]
        # cp = getComponent(n_id, edge_index, idx)
        # cp.visualize()

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        # z.size = (n_id, 100) n_id个节点的表示
        # assoc: 将对应位置上的nid按顺序
        # 例如，assoc长为15，nid=[2,4,6,8,10]，则将assoc[2,4,6,8,10]的位置置为[0,1,2,3,4]

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        y_pred = torch.cat([pos_out], dim=0)

        y_true = []
        for m in msg:
            l = tensor_find(m[16:-16], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        

        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events

def train_gnn_transformer(
    train_data,
    memory,
    gnn,
    neighbor_loader,
    transformer,
    edge_pred,
    optimizer,
    criterion,
    assoc,
    BATCH,
    device
):
    memory.train()
    gnn.train()
    edge_pred.train()
    transformer.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes = set()

    total_loss = 0
    feature_memory = dict()
    sep = torch.zeros(1, 100).to(device)
    train_data = train_data.to(device)
    for batch in train_data.seq_batches(batch_size=BATCH):
        # batch:
        # TemporalData(dst=[1024], msg=[1024, 42], src=[1024], t=[1024])
        # 按照t的顺序取batch
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neighbor_loader.insert(src, pos_dst)
        n_id = torch.cat([src, pos_dst]).unique()
        
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        print(last_update)
        # z.shape = (n_id, 100)
        # last_update.shape = (n_id)
        # last_update = tensor([1522988100361000000, 1522989024614000000, 1522988702903000000,
                        #   0,                   0,                   0,
                        #   0,                   0,                   0,
                        #   0, 1522989025895000000, 1522989001089000000,
                        #   1522988302422000000, 1522989183891000000], device='cuda:7')

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        # z.size = (n_id, 100) n_id个节点的表示
        # assoc: 将对应位置上的nid按顺序
        # 例如，assoc长为15，nid=[2,4,6,8,10]，则将assoc[2,4,6,8,10]的位置置为[0,1,2,3,4]
        sep_dst = list()
        res = list()
        # 遍历所有的边，找到源节点和目标节点的邻接节点，构建新的sequence
        for src1, dst1 in zip(src, pos_dst):
            neighbor1 = edge_index[1][edge_index[0] == assoc[src1]].unique()
            neighbor2 = edge_index[1][edge_index[0] == assoc[dst1]].unique()
            # print(src, neighbor1)
            tmp1 = torch.cat([assoc[src1].unsqueeze(0), neighbor1], dim=-1)
            tmp2 = torch.cat([assoc[dst1].unsqueeze(0), neighbor2], dim=-1)
            neighbor1_feature = z[tmp1]
            neighbor2_feature = z[tmp2]
            sequence = torch.cat([neighbor1_feature, sep, neighbor2_feature], dim=0)

            res.append(sequence)
            sep_dst.append(neighbor1_feature.shape[0])
        
        padded_tensors = nn.utils.rnn.pad_sequence(res, batch_first=True)
        # 获得一个(batch_size, seq_len)的序列 a
        transforer_res = transformer(padded_tensors)
        # transforer_res.shape = (batch_size, seq_len, embedding_size100)
        edge_input = list()
        for i in range(len(sep_dst)):
            edge_input.append(transforer_res[i][sep_dst[i]])
        sep_feature = torch.stack(edge_input).to(device)
        # sep_feature.shape = (batch_size, 100)

        # 提取transformer输出的sep

        pos_out = edge_pred(sep_feature)


        # pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        y_pred = torch.cat([pos_out], dim=0)

        y_true = []
        for m in msg:
            l = tensor_find(m[16:-16], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        
        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        # neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        torch.cuda.empty_cache()

        

        total_loss += float(loss) * batch.num_events
        # print(loss)

    return total_loss / train_data.num_events


def train_transformer(
    train_data,
    transformer,
    link_pred,
    optimizer,
    criterion,
    assoc,
    BATCH,
):
    transformer.train()
    link_pred.train()


    saved_nodes = set()

    total_loss = 0
    for batch in train_data.seq_batches(batch_size=BATCH):
        # batch:
        # TemporalData(dst=[1024], msg=[1024, 42], src=[1024], t=[1024])
        # 按照t的顺序取batch
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        n_id = torch.cat([src, pos_dst]).unique()

        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        num_nodes = BATCH  # 节点数量        
        # 初始化注意力掩码
        attention_mask = torch.full((n_id.size(0), n_id.size(0)), True)  # 默认所有连接为True
        for i in range(n_id.size(0)):
            attention_mask[i, i] = False
        for p_src, p_dst in zip(src, pos_dst):
            # print(p_src, p_dst)
            attention_mask[assoc[p_src], assoc[p_dst]] = False
            # p_src -> p_dst
            # p_dst -> p_src
        # msg = msg.long()
        msg = msg.unique()
        print(msg.shape)

        print(attention_mask)
        node = msg[:,:16].unsqueeze(0)

        z = transformer(node, attention_mask)
        # z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        # z.size = (n_id, 100) n_id个节点的表示
        # assoc: 将对应位置上的nid按顺序
        # 例如，assoc长为15，nid=[2,4,6,8,10]，则将assoc[2,4,6,8,10]的位置置为[0,1,2,3,4]
        # print(z.squeeze(1).shape)
        z = z.squeeze(0)
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        y_pred = torch.cat([pos_out], dim=0)
        print(y_pred)
        y_true = []
        for m in msg:
            l = tensor_find(m[16:-16], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)
        # y_true = (1024)


        loss = criterion(y_pred, y_true)
        

        loss.backward()
        optimizer.step()        

        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events




if __name__ == "__main__":
    
    train_graphs = load_clearscope_e3()
    train_data = train_graphs[-1]
    print(device)
    print(train_data)
    print(artifact_dir)


    # min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    min_dst_idx, max_dst_idx = 0, max_node_num

    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
    # transformer = SparseAttentionTransformer(16, 100, num_heads=4, device=device).to(device)
    transformer = TransformerEncoder(input_dim=16, d_model=100, nhead=4, num_layers=2).to(device)
    # print(sum(p.numel() for p in transformer.parameters()))
    
    print(max_node_num)
    neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)

    memory = TGNMemory(
        max_node_num,
        train_data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(train_data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=train_data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    # link_pred = LinkPredictor(in_channels=embedding_dim, out_channels=train_data.msg.shape[1] - 32).to(device)
    edge_pred = EdgePredictor(in_channels=embedding_dim, out_channels=train_data.msg.shape[1] - 32).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(edge_pred.parameters()) | set(transformer.parameters()),
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, epoch_num+1)):
        for g in train_graphs:
            loss = train_gnn_transformer(
                train_graphs[-1],
                memory,
                gnn,
                neighbor_loader,
                transformer,
                edge_pred,
                optimizer,
                criterion,
                assoc,
                BATCH,
                device
                )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
            model = [memory, gnn, edge_pred, neighbor_loader, transformer]
            torch.save(model, models_dir + "new_models.pt")
    

    # for epoch in tqdm(range(1, 51)):
    #     for g in train_graphs:
    #         loss = train(
    #             g,
    #             memory,
    #             gnn,
    #             link_pred,
    #             neighbor_loader,
    #             optimizer,
    #             criterion,
    #             assoc,
    #             BATCH,
    #         )
    #         logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
    #     #     scheduler.step()
    #         model = [memory, gnn, link_pred, neighbor_loader]
    #         if epoch == 30: 
    #             torch.save(model, "./test_epoch30_emb100/models/models_30.pt")
    #         torch.save(model, "./test_epoch30_emb100/models/models.pt")
