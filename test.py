import logging

from config import *
from model import *
from utils import *

# Setting for logging
logger = logging.getLogger("reconstruction_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'reconstruction2.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

@torch.no_grad()
def test_new(
    inference_data, 
    memory, 
    gnn, 
    transformer, 
    edge_pred, 
    neighbor_loader, 
    nodeid2msg, 
    path, 
    criterion,
    device):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    memory.eval()
    gnn.eval()
    transformer.eval()
    edge_pred.eval()
    
    memory.reset_state()  # Start with a fresh memory. 
    neighbor_loader.reset_state()  # Start with an empty graph.
    
    time_with_loss={}
    total_loss = 0    
    edge_list=[]
    
    unique_nodes=torch.tensor([]).to(device=device)
    total_edges=0


    start_time=inference_data.t[0]
    event_count=0
    
    pos_o=[]
    
    loss_list=[]
    

    print("after merge:",inference_data)
    
    # Record the running time to evaluate the performance
    start = time.perf_counter()
    idx = 0
    inference_data = inference_data.to(device)
    sep = torch.zeros(1, 100).to(device)

    for batch in inference_data.seq_batches(batch_size=BATCH):
        # batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes=torch.cat([unique_nodes,src,pos_dst]).unique()
        total_edges+=BATCH
        
       
        n_id = torch.cat([src, pos_dst]).unique()  
        # print(src, pos_dst)  
        # print(neighbor_loader.neighbors.shape)
        neighbor_loader.insert(src, pos_dst)   
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)

        # cp = getComponent(n_id, edge_index, idx)
        # cp.visualize()
        idx += 1
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

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

        edge_input = list()
        for i in range(len(sep_dst)):
            edge_input.append(transforer_res[i][sep_dst[i]])
        sep_feature = torch.stack(edge_input).to(device)
        # sep_feature.shape = (batch_size, 100)

        # 提取transformer输出的sep

        pos_out = edge_pred(sep_feature)


        # pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        
        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)
#         y_true = torch.cat(
#             [torch.ones(pos_out.size(0))], dim=0).to(torch.long)     
#         y_true=y_true.reshape(-1).to(torch.long)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l) 
        y_true = torch.tensor(y_true).to(device=device)
        y_true=y_true.reshape(-1).to(torch.long).to(device=device)

        # Only consider which edge hasn't been correctly predicted.
        # For benign graphs, the behaviors patterns are similar and therefore their losses are small
        # For anoamlous behaviors, some behaviors might not be seen before, so the probability of predicting those edges are low. Thus their losses are high.
        loss = criterion(y_pred, y_true)

        total_loss += float(loss) * batch.num_events
     
        
        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        # neighbor_loader.insert(src, pos_dst)
        
        # compute the loss for each edge
        each_edge_loss= cal_pos_edges_loss_multiclass(pos_out,y_true,criterion)
        
        for i in range(len(pos_out)):
            srcnode=int(src[i])
            dstnode=int(pos_dst[i])  
            
            srcmsg=str(nodeid2msg[srcnode]) 
            dstmsg=str(nodeid2msg[dstnode])
            t_var=int(t[i])
            edgeindex=tensor_find(msg[i][16:-16],1)   
            edge_type=rel2id[edgeindex]
            loss=each_edge_loss[i]    

            temp_dic={}
            temp_dic['loss']=float(loss)
            temp_dic['srcnode']=srcnode
            temp_dic['dstnode']=dstnode
            temp_dic['srcmsg']=srcmsg
            temp_dic['dstmsg']=dstmsg
            temp_dic['edge_type']=edge_type
            temp_dic['time']=t_var

#             if "netflow" in srcmsg or "netflow" in dstmsg:
#                 temp_dic['loss']=0
            edge_list.append(temp_dic)
        
        event_count+=len(batch.src)
        if t[-1]>start_time+60000000000*15:
            # Here is a checkpoint, which records all edge losses in the current time window
#             loss=total_loss/event_count
            time_interval=ns_time_to_datetime_US(start_time)+"~"+ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval]={'loss':loss,
                                
                                          'nodes_count':len(unique_nodes),
                                          'total_edges':total_edges,
                                          'costed_time':(end-start)}
            
            
            log=open(path+"/"+time_interval+".txt",'w')
            
            for e in edge_list: 
#                 temp_key=e['srcmsg']+e['dstmsg']+e['edge_type']
#                 if temp_key in train_edge_set:      
# #                     e['loss']=(e['loss']-train_edge_set[temp_key]) if e['loss']>=train_edge_set[temp_key] else 0  
# #                     e['loss']=abs(e['loss']-train_edge_set[temp_key])

#                     e['modified']=True
#                 else:
#                     e['modified']=False
                loss+=e['loss']

            loss=loss/event_count   
            logger.info(f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Cost Time: {(end-start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True)   
            for e in edge_list: 
                log.write(str(e))
                log.write("\n") 
            event_count=0
            total_loss=0
            loss=0
            start_time=t[-1]
            log.close()
            edge_list.clear()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            
 
    return time_with_loss

@torch.no_grad()
def test(inference_data, memory, gnn, link_pred, neighbor_loader, nodeid2msg, path, criterion):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    memory.eval()
    gnn.eval()
    link_pred.eval()
    
    memory.reset_state()  # Start with a fresh memory. 
    neighbor_loader.reset_state()  # Start with an empty graph.
    
    time_with_loss={}
    total_loss = 0    
    edge_list=[]
    
    unique_nodes=torch.tensor([]).to(device=device)
    total_edges=0


    start_time=inference_data.t[0]
    event_count=0
    
    pos_o=[]
    
    loss_list=[]
    

    print("after merge:",inference_data)
    
    # Record the running time to evaluate the performance
    start = time.perf_counter()
    idx = 0

    for batch in inference_data.seq_batches(batch_size=BATCH):
        
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes=torch.cat([unique_nodes,src,pos_dst]).unique()
        total_edges+=BATCH
        
       
        n_id = torch.cat([src, pos_dst]).unique()       
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)

        # cp = getComponent(n_id, edge_index, idx)
        # cp.visualize()
        idx += 1
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        
        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)
#         y_true = torch.cat(
#             [torch.ones(pos_out.size(0))], dim=0).to(torch.long)     
#         y_true=y_true.reshape(-1).to(torch.long)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l) 
        y_true = torch.tensor(y_true).to(device=device)
        y_true=y_true.reshape(-1).to(torch.long).to(device=device)

        # Only consider which edge hasn't been correctly predicted.
        # For benign graphs, the behaviors patterns are similar and therefore their losses are small
        # For anoamlous behaviors, some behaviors might not be seen before, so the probability of predicting those edges are low. Thus their losses are high.
        loss = criterion(y_pred, y_true)

        total_loss += float(loss) * batch.num_events
     
        
        # update the edges in the batch to the memory and neighbor_loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
        # compute the loss for each edge
        each_edge_loss= cal_pos_edges_loss_multiclass(pos_out,y_true)
        
        for i in range(len(pos_out)):
            srcnode=int(src[i])
            dstnode=int(pos_dst[i])  
            
            srcmsg=str(nodeid2msg[srcnode]) 
            dstmsg=str(nodeid2msg[dstnode])
            t_var=int(t[i])
            edgeindex=tensor_find(msg[i][16:-16],1)   
            edge_type=rel2id[edgeindex]
            loss=each_edge_loss[i]    

            temp_dic={}
            temp_dic['loss']=float(loss)
            temp_dic['srcnode']=srcnode
            temp_dic['dstnode']=dstnode
            temp_dic['srcmsg']=srcmsg
            temp_dic['dstmsg']=dstmsg
            temp_dic['edge_type']=edge_type
            temp_dic['time']=t_var

#             if "netflow" in srcmsg or "netflow" in dstmsg:
#                 temp_dic['loss']=0
            edge_list.append(temp_dic)
        
        event_count+=len(batch.src)
        if t[-1]>start_time+60000000000*15:
            # Here is a checkpoint, which records all edge losses in the current time window
#             loss=total_loss/event_count
            time_interval=ns_time_to_datetime_US(start_time)+"~"+ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval]={'loss':loss,
                                
                                          'nodes_count':len(unique_nodes),
                                          'total_edges':total_edges,
                                          'costed_time':(end-start)}
            
            
            log=open(path+"/"+time_interval+".txt",'w')
            
            for e in edge_list: 
#                 temp_key=e['srcmsg']+e['dstmsg']+e['edge_type']
#                 if temp_key in train_edge_set:      
# #                     e['loss']=(e['loss']-train_edge_set[temp_key]) if e['loss']>=train_edge_set[temp_key] else 0  
# #                     e['loss']=abs(e['loss']-train_edge_set[temp_key])

#                     e['modified']=True
#                 else:
#                     e['modified']=False
                loss+=e['loss']

            loss=loss/event_count   
            logger.info(f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Cost Time: {(end-start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True)   
            for e in edge_list: 
                log.write(str(e))
                log.write("\n") 
            event_count=0
            total_loss=0
            loss=0
            start_time=t[-1]
            log.close()
            edge_list.clear()
            
 
    return time_with_loss


def load_data(path='/mnt/hdd.data/hzc/clearscopes-e5/train_graphs/'):
    # graph_4_3 - graph_4_5 will be used to initialize node IDF scores.
    graph_5_8 = torch.load(path+"graph_5_8.TemporalData.simple")
    graph_5_9 = torch.load(path+"graph_5_9.TemporalData.simple")
    graph_5_11 = torch.load(path+"graph_5_11.TemporalData.simple")

    # Testing set
    graph_5_12 = torch.load(path+"graph_5_12.TemporalData.simple")
    graph_5_14 = torch.load(path+"graph_5_14.TemporalData.simple")
    graph_5_15 = torch.load(path+"graph_5_15.TemporalData.simple")

    graph_5_17 = torch.load(path+"graph_5_17.TemporalData.simple")


    return [graph_5_8, graph_5_9, graph_5_11, graph_5_12, graph_5_14, graph_5_15, graph_5_17]


if __name__ == "__main__":
    connect = psycopg2.connect(database = database,
                           host = 'localhost',
                           user = 'postgres',
                           password = '123456',
                           port = '5432'
                          )

    cur = connect.cursor()

    sql="select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()

    nodeid2msg={}  # nodeid => msg and node hash => nodeid
    for i in rows:
        nodeid2msg[i[0]]=i[-1]
        nodeid2msg[i[-1]]={i[1]:i[2]}  

    
    criterion = nn.CrossEntropyLoss()
    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
    # graph_5_8, graph_5_9, graph_5_11, graph_5_12, graph_5_14, graph_5_15, graph_5_17 = load_data()
    graph_4_10 = torch.load("/mnt/hdd.data/hzc/clearscopes-e3/train_graphs/graph_4_10.TemporalData.simple")
    graph_4_11 = torch.load("/mnt/hdd.data/hzc/clearscopes-e3/train_graphs/graph_4_11.TemporalData.simple")
    # graph_5_17 = torch.load("/mnt/hdd.data/hzc/clearscopes-e3/train_graphs/graph_5_17.TemporalData.simple")
    memory, gnn, edge_pred, neighbor_loader, transformer = torch.load(f"/mnt/hdd.data/hzc/clearscopes-e3/artifact_cle5_1/models/new_models.pt",map_location=device)
    # print(neighbor_loader.size)
    neighbor_loader = LastNeighborLoader(max_node_num, size=20, device=device)

    # Reconstruct the edges in each day
    # test(inference_data=graph_5_8,
    #      memory=memory,
    #      gnn=gnn,
    #      link_pred=link_pred,
    #      neighbor_loader=neighbor_loader,
    #      nodeid2msg=nodeid2msg,
    #      path=artifact_dir + "graph_5_8",
    #      criterion=criterion)
    
    # test(inference_data=graph_5_9,
    #      memory=memory,
    #      gnn=gnn,
    #      link_pred=link_pred,
    #      neighbor_loader=neighbor_loader,
    #      nodeid2msg=nodeid2msg,
    #      path=artifact_dir + "graph_5_9",
    #      criterion=criterion)

    # test(inference_data=graph_5_11,
    #      memory=memory,
    #      gnn=gnn,
    #      link_pred=link_pred,
    #      neighbor_loader=neighbor_loader,
    #      nodeid2msg=nodeid2msg,
    #      path=artifact_dir + "graph_5_11",
    #      criterion=criterion)

    # test(inference_data=graph_5_12,
    #      memory=memory,
    #      gnn=gnn,
    #      link_pred=link_pred,
    #      neighbor_loader=neighbor_loader,
    #      nodeid2msg=nodeid2msg,
    #      path=artifact_dir + "graph_5_12",
    #      criterion=criterion)

    # test_new(inference_data=graph_5_14,
    #      memory=memory,
    #      gnn=gnn,
    #      edge_pred=edge_pred,
    #      transformer=transformer,
    #      neighbor_loader=neighbor_loader,
    #      nodeid2msg=nodeid2msg,
    #      path=artifact_dir + "graph_5_14",
    #      criterion=criterion,
    #      device=device)

    test_new(inference_data=graph_4_10,
         memory=memory,
         gnn=gnn,
         edge_pred=edge_pred,
         transformer=transformer,
         neighbor_loader=neighbor_loader,
         nodeid2msg=nodeid2msg,
         path=artifact_dir + "graph_4_10",
         criterion=criterion,
         device=device)
    
    test_new(inference_data=graph_4_11,
        memory=memory,
        gnn=gnn,
        edge_pred=edge_pred,
        transformer=transformer,
        neighbor_loader=neighbor_loader,
        nodeid2msg=nodeid2msg,
        path=artifact_dir + "graph_4_11",
        criterion=criterion,
        device=device)

    # test_new(inference_data=graph_5_17,
    #     memory=memory,
    #     gnn=gnn,
    #     edge_pred=edge_pred,
    #     transformer=transformer,
    #     neighbor_loader=neighbor_loader,
    #     nodeid2msg=nodeid2msg,
    #     path=artifact_dir + "graph_5_17",
    #     criterion=criterion,
    #     device=device)

    # test_new(inference_data=graph_5_17,
    #      memory=memory,
    #      gnn=gnn,
    #      edge_pred=edge_pred,
    #      transformer=transformer,
    #      neighbor_loader=neighbor_loader,
    #      nodeid2msg=nodeid2msg,
    #      path=artifact_dir + "graph_5_17",
    #      criterion=criterion,
    #      device=device)
