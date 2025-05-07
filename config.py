########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
raw_dir = "/mnt/hdd.data/hzc/clearscopes-e3"

# The directory to save all artifacts
artifact_dir = "/mnt/hdd.data/hzc/theia-e5/version3-1/"

# The directory to save the vectorized graphs
graphs_dir = artifact_dir + "graphs/"

# The directory to save the models
models_dir = artifact_dir + "models/"

# The directory to save the results after testing
test_re = artifact_dir + "test_re/"

# The directory to save all visualized results
vis_re = artifact_dir + "vis_re/"



########################################################
#
#               Database settings
#
########################################################

# Database name
database = 'tc_e5_theia_dataset_db'

#             Name             |  Owner   | Encoding |   Collate   |    Ctype    |   Access privileges   
# -----------------------------+----------+----------+-------------+-------------+-----------------------
#  optc_db                     | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  postgres                    | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  streamspot                  | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_cadet_dataset_db         | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_clearscope3_dataset_db   | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_e5_cadets_dataset_db     | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_e5_clearscope_dataset_db | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_e5_theia_dataset_db      | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_theia_dataset_db         | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
host = 'localhost'

# Database user
user = 'postgres'

# The password to the database user
password = '123456'

# The port number for Postgres
port = '5432'


########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
# edge_reversed = [
#     "EVENT_ACCEPT",
#     "EVENT_RECVFROM",
#     "EVENT_RECVMSG"
# ]

# # The following edges are the types only considered to construct the
# # temporal graph for experiments.
# include_edge_type=[
#     "EVENT_WRITE",
#     "EVENT_READ",
#     "EVENT_CLOSE",
#     "EVENT_OPEN",
#     "EVENT_EXECUTE",
#     "EVENT_SENDTO",
#     "EVENT_RECVFROM",
# ]

# The map between edge type and edge ID
# rel2id for cadets
rel2id_cadets3 = {
    1: "EVENT_ACCEPT",
    "EVENT_ACCEPT": 1,
    2: "EVENT_CLONE",
    "EVENT_CLONE": 2,
    3: "EVENT_CLOSE",
    "EVENT_CLOSE": 3,
    4: "EVENT_CREATE_OBJECT",
    "EVENT_CREATE_OBJECT": 4,
    5: "EVENT_EXECUTE",
    "EVENT_EXECUTE": 5,
    6: "EVENT_OPEN",
    "EVENT_OPEN": 6,
    7: "EVENT_READ",
    "EVENT_READ": 7,
    8: "EVENT_RECVFROM",
    "EVENT_RECVFROM": 8,
    9: "EVENT_SENDTO",
    "EVENT_SENDTO": 9,
    10: "EVENT_WRITE",
    "EVENT_WRITE": 10,
}

# rel2id for cadets_5
rel2id_cadets5 = {
    1: 'EVENT_CLOSE',
    'EVENT_CLOSE': 1,
    2: 'EVENT_OPEN',
    'EVENT_OPEN': 2,
    3: 'EVENT_READ',
    'EVENT_READ': 3,
    4: 'EVENT_WRITE',
    'EVENT_WRITE': 4,
    5: 'EVENT_EXECUTE',
    'EVENT_EXECUTE': 5,
    6: 'EVENT_RECVFROM',
    'EVENT_RECVFROM': 6,
    7: 'EVENT_RECVMSG',
    'EVENT_RECVMSG': 7,
    8: 'EVENT_SENDMSG',
    'EVENT_SENDMSG': 8,
    9: 'EVENT_SENDTO',
    'EVENT_SENDTO': 9
}

# rel id for theia
rel2id_theia = {
    1: 'EVENT_CONNECT',
    'EVENT_CONNECT': 1,
    2: 'EVENT_EXECUTE',
    'EVENT_EXECUTE': 2,
    3: 'EVENT_OPEN',
    'EVENT_OPEN': 3,
    4: 'EVENT_READ',
    'EVENT_READ': 4,
    5: 'EVENT_RECVFROM',
    'EVENT_RECVFROM': 5,
    6: 'EVENT_RECVMSG',
    'EVENT_RECVMSG': 6,
    7: 'EVENT_SENDMSG',
    'EVENT_SENDMSG': 7,
    8: 'EVENT_SENDTO',
    'EVENT_SENDTO': 8,
    9: 'EVENT_WRITE',
    'EVENT_WRITE': 9
}

rel2id_clearscope3 = {
    1: 'EVENT_CLOSE',
    'EVENT_CLOSE': 1,
    2: 'EVENT_OPEN',
    'EVENT_OPEN': 2,
    3: 'EVENT_READ',
    'EVENT_READ': 3,
    4: 'EVENT_WRITE',
    'EVENT_WRITE': 4,
    5: 'EVENT_RECVFROM',
    'EVENT_RECVFROM': 5,
    6: 'EVENT_RECVMSG',
    'EVENT_RECVMSG': 6,
    7: 'EVENT_SENDMSG',
    'EVENT_SENDMSG': 7,
    8: 'EVENT_SENDTO',
    'EVENT_SENDTO': 8
}

rel2id_clearscope5 = {
    1: 'EVENT_ACCEPT',
    'EVENT_ACCEPT': 1,
    2: 'EVENT_CLONE',
    'EVENT_CLONE': 2,
    3: 'EVENT_CLOSE',
    'EVENT_CLOSE': 3,
    4: 'EVENT_CREATE_OBJECT',
    'EVENT_CREATE_OBJECT': 4,
    5: 'EVENT_EXECUTE',
    'EVENT_EXECUTE': 5,
    6: 'EVENT_OPEN',
    'EVENT_OPEN': 6,
    7: 'EVENT_READ',
    'EVENT_READ': 7,
    8: 'EVENT_RECVFROM',
    'EVENT_RECVFROM': 8,
    9: 'EVENT_SENDTO',
    'EVENT_SENDTO': 9,
    10: 'EVENT_WRITE',
    'EVENT_WRITE': 10
}

rel2id = rel2id_theia

########################################################
#
#                   Model dimensionality
#
########################################################

# Node Embedding Dimension
node_embedding_dim = 16

# Node State Dimension
node_state_dim = 100

memory_dim = 100

# Neighborhood Sampling Size
neighbor_size = 20

# Edge Embedding Dimension
edge_dim = 100

# The time encoding Dimension
time_dim = 100

embedding_dim = 100

max_node_num_theia_e3 = 828398 # theia-e3
max_node_num_theia_e5 = 967389 # theia-e5
max_node_num_cadets_e5 = 262626 # cadets-e5
max_node_num_cadets_e3 = 268243 # cadets-e3
max_node_num_clearscope_e3 = 172724 # clearscope-e3
max_node_num_clearscope_e5 = 139961 # clearscope-e5

max_node_num = max_node_num_theia_e5


########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 128
device = 'cuda:2'

# Parameters for optimizer
lr=0.00005
eps=1e-08
weight_decay=0.01

epoch_num=50

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 15


########################################################
#
#                   Threshold
#
########################################################

beta_day6 = 100
beta_day7 = 100
