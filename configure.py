train_cand_size = 20

dev_cand_size = 20

test_cand_size = 1000

# in the future version, the word idx should be directed to the model related dir
word_idx_file_path = "data/word.idx"
pred_file_path = "data/pred.txt"
# label_idx_file_path = "data/label.idx"

train_batch_size = 32
dev_batch_size = 20
test_batch_size = 100

l2_lambda = 0.000001
learning_rate = 0.001
epochs = 10
encoder_hidden_dim = 200
word_size_max = 1

dropout = 0.0

node_vec_method = "lstm" # lstm or word_emb

# path_embed_method = "lstm" # cnn or lstm or bi-lstm

unknown_word = "**UNK**"
deal_unknown_words = True

if_use_pretrained_embedding = True
pretrained_word_embedding_dim = 300
pretrained_word_embedding_path = "DBP15K/sub.glove.300d"
word_embedding_dim = 100

num_layers = 1 # 1 or 2

# the following are for the graph encoding method
weight_decay = 0.0000
sample_size_per_layer = 1
hidden_layer_dim = 100
feature_max_len = 1
feature_encode_type = "uni"
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
graph_encode_direction = "bi" # "single" or "bi"

concat = True

encoder = "gated_gcn" # "gated_gcn" "gcn"  "seq"

lstm_in_gcn = "none" # before, after, none

aggregator_dim_first = 100
aggregator_dim_second = 100
gcn_window_size_first = 1
gcn_window_size_second = 2
gcn_layer_size_first = 1
gcn_layer_size_second = 1

with_match_highway = False
with_gcn_highway = False
if_use_multiple_gcn_1_state = False
if_use_multiple_gcn_2_state = False

agg_sim_method = "GCN" # "GCN" or LSTM

gcn_type_first = 'mean_pooling' # GCN, max_pooling, mean_pooling, lstm, att
gcn_type_second = 'mean_pooling'

cosine_MP_dim = 10

pred_method = "graph_level"  # graph_level or node_level


