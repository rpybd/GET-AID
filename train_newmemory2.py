import logging

from config import *
from data import *
from model import *
from new_memory import NewMemory

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
    gnn.train()
    edge_pred.train()
    transformer.train()

    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    sep = torch.zeros(1, 100).to(device)
    train_data = train_data.to(device)
    for batch in train_data.seq_batches(batch_size=BATCH):
        # batch:
        # TemporalData(dst=[1024], msg=[1024, 42], src=[1024], t=[1024])
        # 按照t的顺序取batch
        optimizer.zero_grad()
        batch_size = batch.src.shape[0]
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        # neighbor_loader.insert(src, pos_dst)
        n_id = torch.cat([src, pos_dst]).unique()

        
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = z.to(device)
        last_update = last_update.to(device)
        cp = getComponent(n_id, edge_index)
        tmp = cp.find_connected_components()
        
        # n_id_mask = [[1 for _ in range((max_value) + 1)] for _ in range((max_value) + 1)]
        # print(tmp)
        # print(len(n_id_mask))
        # for component in tmp:
        #     for node_id in component:
        #         for n in component:
        #             n_id_mask[assoc[node_id]][n] = 0
        # 将非联通域的节点之间添加 mask
        # 假如连通域是 [[0, 1], [2], [3]]
        # 则 mask 为 [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]        
        
        # z.shape = (n_id, 100)
        # last_update.shape = (n_id)
        # last_update = tensor(
                        # [1522988100361000000, 1522989024614000000, 1522988702903000000,
                        #   0,                   0,                   0,
                        #   0,                   0,                   0,
                        #   0,                  1522989025895000000, 1522989001089000000])

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        # z.size = (n_id, 100) n_id个节点的表示
        # assoc: 将对应位置上的nid按顺序
        # 例如，assoc长为15，nid=[2,4,6,8,10]，则将assoc[2,4,6,8,10]的位置置为[0,1,2,3,4]

        src1_feature, src2_feature = list(), list()
        # mask_sep = [[1 for _ in range(len(n_id) + BATCH)] for _ in range(BATCH)]
        mask = torch.ones(len(n_id) + batch_size, len(n_id) + batch_size, dtype=bool)
        for component in tmp:
            for first_node in component:
                for second_node in component:
                    mask[first_node, second_node] = 0
        # 处理 sep 和节点之间的 mask
        for idx, (src1, dst1) in enumerate(zip(src, pos_dst), start=len(n_id)):
            # 对源节点、目标节点和它们的邻接节点全部添加 mask
            neighbor1 = edge_index[1][edge_index[0] == assoc[src1]].unique()
            neighbor2 = edge_index[1][edge_index[0] == assoc[dst1]].unique()
            #
            mask[idx, assoc[src1]]= 0
            mask[idx, assoc[dst1]] = 0
            for neighbor in neighbor1:
                if neighbor:
                    mask[idx, neighbor] = 0
            for neighbor in neighbor2:
                if neighbor:
                    mask[idx, neighbor] = 0
            mask[idx, idx] = 0
        # n_id_mask: 记录节点间的 mask
        # sep_mask: 记录 sep 和节点间的 mask
        # mask = [[0 for _ in range(len(n_id) + BATCH)] for _ in range(len(n_id) + BATCH)]
      
        transformer_res = transformer(z, mask, batch_size)
        
        
        sep_feature = transformer_res[:, -batch_size:, :].squeeze(0)
        node_embedding = transformer_res[:, :-batch_size, :].squeeze(0)
        
        memory.update2(n_id, node_embedding.detach(), src, pos_dst, t)

        # 提取transformer输出的sep和node embedding

        pos_out = edge_pred(sep_feature)

        y_pred = torch.cat([pos_out], dim=0)

        y_true = []
        for m in msg:
            l = tensor_find(m[16:-16], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)
        
        # Update memory and neighbor loader with ground-truth state.
        # memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        total_loss += float(loss) * batch.num_events
        # print(loss)

    return total_loss / train_data.num_events


if __name__ == "__main__":
    
    train_graphs = load_cadets_e3()
    train_data = train_graphs[-1]
    print(device)
    print(train_data)
    print(artifact_dir)


    # min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    min_dst_idx, max_dst_idx = 0, max_node_num

    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
    # transformer = SparseAttentionTransformer(16, 100, num_heads=4, device=device).to(device)
    transformer = TransformerEncoder2(input_dim=16, d_model=100, nhead=4, num_layers=2).to(device)
    # print(sum(p.numel() for p in transformer.parameters()))
    
    print(max_node_num)
    neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)

    memory = NewMemory(num_nodes=max_node_num, memory_dim=memory_dim)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=train_data.msg.size(-1),
        time_enc=TimeEncoder(time_dim),
    ).to(device)

    # link_pred = LinkPredictor(in_channels=embedding_dim, out_channels=train_data.msg.shape[1] - 32).to(device)
    edge_pred = EdgePredictor(in_channels=embedding_dim, out_channels=train_data.msg.shape[1] - 32).to(device)

    optimizer = torch.optim.Adam(
        set(gnn.parameters()) | set(edge_pred.parameters()) | set(transformer.parameters()),
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, epoch_num+1)):
        for g in train_graphs:
            loss = train_gnn_transformer(
                g,
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
            torch.save(model, models_dir + f"new_models+{epoch}.pt")
            