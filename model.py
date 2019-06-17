import tensorflow as tf
from neigh_samplers import UniformNeighborSampler
from aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator, GCNAggregator, SeqAggregator, AttentionAggregator
from graph_match_utils import match_graph_1_with_graph_2
from matching_model_options import options
from match_utils import multi_highway_layer
import numpy as np
import layer_utils

class GraphMatchNN(object):

    def __init__(self, mode, conf, pretrained_word_embeddings):

        self.mode = mode
        self.word_vocab_size = conf.word_vocab_size
        self.l2_lambda = conf.l2_lambda
        self.word_embedding_dim = conf.hidden_layer_dim
        self.encoder_hidden_dim = conf.encoder_hidden_dim


        # the setting for the GCN
        self.num_layers = conf.num_layers
        self.graph_encode_direction = conf.graph_encode_direction
        self.hidden_layer_dim = conf.hidden_layer_dim
        self.concat = conf.concat

        self.y_true = tf.placeholder(tf.float32, [None, 2], name="true_labels")

        # the following place holders are for the first graph
        self.fw_adj_info_first = tf.placeholder(tf.int32, [None, None])               # the fw adj info for each node
        self.bw_adj_info_first = tf.placeholder(tf.int32, [None, None])               # the bw adj info for each node
        self.feature_info_first = tf.placeholder(tf.int32, [None, None])              # the feature info for each node
        self.feature_len_first = tf.placeholder(tf.int32, [None])                     # the feature len for each node
        self.batch_nodes_first = tf.placeholder(tf.int32, [None, None])               # the nodes for the first batch
        self.batch_mask_first = tf.placeholder(tf.float32, [None, None])              # the mask for the first batch
        self.looking_table_first = tf.placeholder(tf.int32, [None])                   # the looking table for the first batch
        self.entity_index_first = tf.placeholder(tf.int32, [None])                    # the entity node index in each graph

        self.fw_adj_info_second = tf.placeholder(tf.int32, [None, None])              # the fw adj info for each node
        self.bw_adj_info_second = tf.placeholder(tf.int32, [None, None])              # the bw adj info for each node
        self.feature_info_second = tf.placeholder(tf.int32, [None, None])             # the feature info for each node
        self.feature_len_second = tf.placeholder(tf.int32, [None])                    # the feature len for each node
        self.batch_nodes_second = tf.placeholder(tf.int32, [None, None])              # the nodes for the first batch
        self.batch_mask_second = tf.placeholder(tf.float32, [None, None])             # the mask for the second batch
        self.looking_table_second = tf.placeholder(tf.int32, [None])                  # the looking table for the second batch
        self.entity_index_second = tf.placeholder(tf.int32, [None])                   # the entity node index in each graph

        self.with_match_highway = conf.with_match_highway
        self.with_gcn_highway = conf.with_gcn_highway
        self.if_use_multiple_gcn_1_state = conf.if_use_multiple_gcn_1_state
        self.if_use_multiple_gcn_2_state = conf.if_use_multiple_gcn_2_state

        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.pretrained_word_size = conf.pretrained_word_size
        self.learned_word_size = conf.learned_word_size

        self.sample_size_per_layer_first = tf.shape(self.fw_adj_info_first)[1]
        self.sample_size_per_layer_second = tf.shape(self.fw_adj_info_second)[1]
        self.batch_size = tf.shape(self.y_true)[0]
        self.dropout = conf.dropout

        self.fw_aggregators_first = []
        self.bw_aggregators_first = []
        self.aggregator_dim_first = conf.aggregator_dim_first
        self.gcn_window_size_first = conf.gcn_window_size_first
        self.gcn_layer_size_first = conf.gcn_layer_size_first

        self.fw_aggregators_second = []
        self.bw_aggregators_second = []
        self.aggregator_dim_second = conf.aggregator_dim_second
        self.gcn_window_size_second = conf.gcn_window_size_second
        self.gcn_layer_size_second = conf.gcn_layer_size_second

        self.if_pred_on_dev = False
        self.learning_rate = conf.learning_rate

        self.agg_sim_method = conf.agg_sim_method

        self.agg_type_first = conf.gcn_type_first
        self.agg_type_second = conf.gcn_type_second

        options['cosine_MP_dim'] = conf.cosine_MP_dim

        self.node_vec_method = conf.node_vec_method
        self.pred_method = conf.pred_method
        self.watch = {}

    def _build_graph(self):
        node_1_mask = self.batch_mask_first
        node_2_mask = self.batch_mask_second
        node_1_looking_table = self.looking_table_first
        node_2_looking_table = self.looking_table_second

        node_2_aware_representations = []
        node_2_aware_dim = 0
        node_1_aware_representations = []
        node_1_aware_dim = 0

        pad_word_embedding = tf.zeros([1, self.word_embedding_dim])  # this is for the PAD symbol
        self.word_embeddings = tf.concat([pad_word_embedding,
                                          tf.get_variable('pretrained_embedding', shape=[self.pretrained_word_size, self.word_embedding_dim],
                                                          initializer=tf.constant_initializer(self.pretrained_word_embeddings), trainable=True),
                                          tf.get_variable('W_train',
                                                          shape=[self.learned_word_size, self.word_embedding_dim],
                                                          initializer=tf.contrib.layers.xavier_initializer(),
                                                          trainable=True)], 0)

        self.watch['word_embeddings'] = self.word_embeddings

        # ============ encode node feature by looking up word embedding =============
        with tf.variable_scope('node_rep_gen'):
            # [node_size, hidden_layer_dim]
            feature_embedded_chars_first = tf.nn.embedding_lookup(self.word_embeddings, self.feature_info_first)
            graph_1_size = tf.shape(feature_embedded_chars_first)[0]

            feature_embedded_chars_second = tf.nn.embedding_lookup(self.word_embeddings, self.feature_info_second)
            graph_2_size = tf.shape(feature_embedded_chars_second)[0]

            if self.node_vec_method == "lstm":
                cell = self.build_encoder_cell(1, self.hidden_layer_dim)

                outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=feature_embedded_chars_first,
                                                           sequence_length=self.feature_len_first, dtype=tf.float32)
                node_1_rep = layer_utils.collect_final_step_of_lstm(outputs, self.feature_len_first-1)

                outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=feature_embedded_chars_second,
                                                           sequence_length=self.feature_len_second, dtype=tf.float32)
                node_2_rep = layer_utils.collect_final_step_of_lstm(outputs, self.feature_len_second-1)

            elif self.node_vec_method == "word_emb":
                node_1_rep = tf.reshape(feature_embedded_chars_first, [graph_1_size, -1])
                node_2_rep = tf.reshape(feature_embedded_chars_second, [graph_2_size, -1])

            self.watch["node_1_rep_initial"] = node_1_rep

        # ============ encode node feature by GCN =============
        with tf.variable_scope('first_gcn') as first_gcn_scope:
            # shape of node embedding: [batch_size, single_graph_nodes_size, node_embedding_dim]
            # shape of node size: [batch_size]
            gcn_1_res = self.gcn_encode(self.batch_nodes_first,
                                        node_1_rep,
                                        self.fw_adj_info_first, self.bw_adj_info_first,
                                        input_node_dim=self.word_embedding_dim,
                                        output_node_dim=self.aggregator_dim_first,
                                        fw_aggregators=self.fw_aggregators_first,
                                        bw_aggregators=self.bw_aggregators_first,
                                        window_size=self.gcn_window_size_first,
                                        layer_size=self.gcn_layer_size_first,
                                        scope="first_gcn",
                                        agg_type=self.agg_type_first,
                                        sample_size_per_layer=self.sample_size_per_layer_first,
                                        keep_inter_state=self.if_use_multiple_gcn_1_state)

            node_1_rep = gcn_1_res[0]
            node_1_rep_dim = gcn_1_res[3]

            gcn_2_res = self.gcn_encode(self.batch_nodes_second,
                                        node_2_rep,
                                        self.fw_adj_info_second,
                                        self.bw_adj_info_second,
                                        input_node_dim=self.word_embedding_dim,
                                        output_node_dim=self.aggregator_dim_first,
                                        fw_aggregators=self.fw_aggregators_first,
                                        bw_aggregators=self.bw_aggregators_first,
                                        window_size=self.gcn_window_size_first,
                                        layer_size=self.gcn_layer_size_first,
                                        scope="first_gcn",
                                        agg_type=self.agg_type_first,
                                        sample_size_per_layer=self.sample_size_per_layer_second,
                                        keep_inter_state=self.if_use_multiple_gcn_1_state)

            node_2_rep = gcn_2_res[0]
            node_2_rep_dim = gcn_2_res[3]

        self.watch["node_1_rep_first_GCN"] = node_1_rep
        self.watch["node_1_mask"] = node_1_mask

        # mask
        node_1_rep = tf.multiply(node_1_rep, tf.expand_dims(node_1_mask, 2))
        node_2_rep = tf.multiply(node_2_rep, tf.expand_dims(node_2_mask, 2))

        self.watch["node_1_rep_first_GCN_masked"] = node_1_rep

        if self.pred_method == "node_level":
            entity_1_rep = tf.reshape(tf.nn.embedding_lookup(tf.transpose(node_1_rep, [1, 0, 2]), tf.constant(0)), [-1, node_1_rep_dim])
            entity_2_rep = tf.reshape(tf.nn.embedding_lookup(tf.transpose(node_2_rep, [1, 0, 2]), tf.constant(0)), [-1, node_2_rep_dim])

            entity_1_2_diff = entity_1_rep - entity_2_rep
            entity_1_2_sim = entity_1_rep * entity_2_rep

            aggregation = tf.concat([entity_1_rep, entity_2_rep, entity_1_2_diff, entity_1_2_sim], axis=1)
            aggregation_dim = 4 * node_1_rep_dim

            w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [aggregation_dim / 2, 2], dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

            # ====== Prediction Layer ===============
            logits = tf.matmul(aggregation, w_0) + b_0
            logits = tf.tanh(logits)
            logits = tf.matmul(logits, w_1) + b_1

        elif self.pred_method == "graph_level":
            # if the prediction method is graph_level, we perform the graph matching based prediction

            assert node_1_rep_dim == node_2_rep_dim
            input_dim = node_1_rep_dim

            with tf.variable_scope('node_level_matching') as matching_scope:
                # ========= node level matching ===============
                (match_reps, match_dim) = match_graph_1_with_graph_2(node_1_rep, node_2_rep, node_1_mask, node_2_mask, input_dim,
                                                                     options=options, watch=self.watch)

                matching_scope.reuse_variables()

                node_2_aware_representations.append(match_reps)
                node_2_aware_dim += match_dim

                (match_reps, match_dim) = match_graph_1_with_graph_2(node_2_rep, node_1_rep, node_2_mask, node_1_mask, input_dim,
                                                                     options=options, watch=self.watch)

                node_1_aware_representations.append(match_reps)
                node_1_aware_dim += match_dim

            # TODO: add one more MP matching over the graph representation
            # with tf.variable_scope('context_MP_matching'):
            #     for i in range(options['context_layer_num']):
            #         with tf.variable_scope('layer-{}',format(i)):

            # [batch_size, single_graph_nodes_size, node_2_aware_dim]
            node_2_aware_representations = tf.concat(axis=2, values=node_2_aware_representations)

            # [batch_size, single_graph_nodes_size, node_1_aware_dim]
            node_1_aware_representations = tf.concat(axis=2, values=node_1_aware_representations)

            # if self.mode == "train":
            #     node_2_aware_representations = tf.nn.dropout(node_2_aware_representations, (1 - options['dropout_rate']))
            #     node_1_aware_representations = tf.nn.dropout(node_1_aware_representations, (1 - options['dropout_rate']))

            # ========= Highway layer ==============
            if self.with_match_highway:
                with tf.variable_scope("left_matching_highway"):
                    node_2_aware_representations = multi_highway_layer(node_2_aware_representations, node_2_aware_dim,
                                                                        options['highway_layer_num'])
                with tf.variable_scope("right_matching_highway"):
                    node_1_aware_representations = multi_highway_layer(node_1_aware_representations, node_1_aware_dim,
                                                                       options['highway_layer_num'])

            self.watch["node_1_rep_match"] = node_2_aware_representations

            # ========= Aggregation Layer ==============
            aggregation_representation = []
            aggregation_dim = 0

            node_2_aware_aggregation_input = node_2_aware_representations
            node_1_aware_aggregation_input = node_1_aware_representations

            self.watch["node_1_rep_match_layer"] = node_2_aware_aggregation_input

            with tf.variable_scope('aggregation_layer'):
                # TODO: now we only have 1 aggregation layer; need to change this part if support more aggregation layers
                # [batch_size, single_graph_nodes_size, node_2_aware_dim]
                node_2_aware_aggregation_input = tf.multiply(node_2_aware_aggregation_input,
                                                             tf.expand_dims(node_1_mask, axis=-1))

                # [batch_size, single_graph_nodes_size, node_1_aware_dim]
                node_1_aware_aggregation_input = tf.multiply(node_1_aware_aggregation_input,
                                                             tf.expand_dims(node_2_mask, axis=-1))

                if self.agg_sim_method == "GCN":
                    # [batch_size*single_graph_nodes_size, node_2_aware_dim]
                    node_2_aware_aggregation_input = tf.reshape(node_2_aware_aggregation_input,
                                                                shape=[-1, node_2_aware_dim])

                    # [batch_size*single_graph_nodes_size, node_1_aware_dim]
                    node_1_aware_aggregation_input = tf.reshape(node_1_aware_aggregation_input,
                                                                shape=[-1, node_1_aware_dim])

                    # [node_1_size, node_2_aware_dim]
                    node_1_rep = tf.concat([tf.nn.embedding_lookup(node_2_aware_aggregation_input, node_1_looking_table),
                                            tf.zeros([1, node_2_aware_dim])], 0)

                    # [node_2_size, node_1_aware_dim]
                    node_2_rep = tf.concat([tf.nn.embedding_lookup(node_1_aware_aggregation_input, node_2_looking_table),
                                            tf.zeros([1, node_1_aware_dim])], 0)

                    gcn_1_res = self.gcn_encode(self.batch_nodes_first,
                                                node_1_rep,
                                                self.fw_adj_info_first,
                                                self.bw_adj_info_first,
                                                input_node_dim=node_2_aware_dim,
                                                output_node_dim=self.aggregator_dim_second,
                                                fw_aggregators=self.fw_aggregators_second,
                                                bw_aggregators=self.bw_aggregators_second,
                                                window_size=self.gcn_window_size_second,
                                                layer_size=self.gcn_layer_size_second,
                                                scope="second_gcn",
                                                agg_type=self.agg_type_second,
                                                sample_size_per_layer=self.sample_size_per_layer_first,
                                                keep_inter_state=self.if_use_multiple_gcn_2_state)

                    max_graph_1_rep = gcn_1_res[1]
                    mean_graph_1_rep = gcn_1_res[2]
                    graph_1_rep_dim = gcn_1_res[3]

                    gcn_2_res = self.gcn_encode(self.batch_nodes_second,
                                                node_2_rep,
                                                self.fw_adj_info_second,
                                                self.bw_adj_info_second,
                                                input_node_dim=node_1_aware_dim,
                                                output_node_dim=self.aggregator_dim_second,
                                                fw_aggregators=self.fw_aggregators_second,
                                                bw_aggregators=self.bw_aggregators_second,
                                                window_size=self.gcn_window_size_second,
                                                layer_size=self.gcn_layer_size_second,
                                                scope="second_gcn",
                                                agg_type=self.agg_type_second,
                                                sample_size_per_layer=self.sample_size_per_layer_second,
                                                keep_inter_state=self.if_use_multiple_gcn_2_state)

                    max_graph_2_rep = gcn_2_res[1]
                    mean_graph_2_rep = gcn_2_res[2]
                    graph_2_rep_dim = gcn_2_res[3]

                    assert graph_1_rep_dim == graph_2_rep_dim

                    if self.if_use_multiple_gcn_2_state:
                        graph_1_reps = gcn_1_res[5]
                        graph_2_reps = gcn_2_res[5]
                        inter_dims = gcn_1_res[6]
                        for idx in range(len(graph_1_reps)):
                            (max_graph_1_rep_tmp, mean_graph_1_rep_tmp) = graph_1_reps[idx]
                            (max_graph_2_rep_tmp, mean_graph_2_rep_tmp) = graph_2_reps[idx]
                            inter_dim = inter_dims[idx]
                            aggregation_representation.append(max_graph_1_rep_tmp)
                            aggregation_representation.append(mean_graph_1_rep_tmp)
                            aggregation_representation.append(max_graph_2_rep_tmp)
                            aggregation_representation.append(mean_graph_2_rep_tmp)
                            aggregation_dim += 4 * inter_dim

                    else:
                        aggregation_representation.append(max_graph_1_rep)
                        aggregation_representation.append(mean_graph_1_rep)
                        aggregation_representation.append(max_graph_2_rep)
                        aggregation_representation.append(mean_graph_2_rep)
                        aggregation_dim = 4 * graph_1_rep_dim

                    # aggregation_representation = tf.concat(aggregation_representation, axis=1)

                    gcn_2_window_size = int(len(aggregation_representation)/4)
                    aggregation_dim = aggregation_dim/gcn_2_window_size

                    w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
                    b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)
                    w_1 = tf.get_variable("w_1", [aggregation_dim / 2, 2], dtype=tf.float32)
                    b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

                    weights = tf.get_variable("gcn_2_window_weights", [gcn_2_window_size], dtype=tf.float32)

                    # shape: [gcn_2_window_size, batch_size, 2]
                    logits = []
                    for layer_idx in range(gcn_2_window_size):
                        max_graph_1_rep = aggregation_representation[layer_idx * 4 + 0]
                        mean_graph_1_rep = aggregation_representation[layer_idx * 4 + 1]
                        max_graph_2_rep = aggregation_representation[layer_idx * 4 + 2]
                        mean_graph_2_rep = aggregation_representation[layer_idx * 4 + 3]

                        aggregation_representation_single = tf.concat([max_graph_1_rep, mean_graph_1_rep, max_graph_2_rep, mean_graph_2_rep], axis=1)

                        # ====== Prediction Layer ===============
                        logit = tf.matmul(aggregation_representation_single, w_0) + b_0
                        logit = tf.tanh(logit)
                        logit = tf.matmul(logit, w_1) + b_1
                        logits.append(logit)

                    if len(logits) != 1:
                        logits = tf.reshape(tf.concat(logits, axis=0), [gcn_2_window_size, -1, 2])
                        logits = tf.transpose(logits, [1, 0, 2])
                        logits = tf.multiply(logits, tf.expand_dims(weights, axis=-1))
                        logits = tf.reduce_sum(logits, axis=1)
                    else:
                        logits = tf.reshape(logits, [-1, 2])



        # ====== Highway layer ============
        # if options['with_aggregation_highway']:

        with tf.name_scope("loss"):
            self.y_pred = tf.nn.softmax(logits)
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits, name="xentropy_loss")) / tf.cast(self.batch_size, tf.float32)

        # ============  Training Objective ===========================
        if self.mode == "train" and not self.if_pred_on_dev:
            optimizer = tf.train.AdamOptimizer()
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
            self.training_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def build_encoder_cell(self, num_layers, hidden_size):
        if num_layers == 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            if self.mode == "train" and self.dropout > 0.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout)
            return cell
        else:
            cell_list = []
            for i in range(num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                if self.mode == "train" and self.dropout > 0.0:
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, self.dropout)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def gcn_encode(self, batch_nodes, embedded_node_rep, fw_adj_info, bw_adj_info, input_node_dim, output_node_dim, fw_aggregators, bw_aggregators, window_size, layer_size, scope, agg_type, sample_size_per_layer, keep_inter_state=False):
        with tf.variable_scope(scope):
            single_graph_nodes_size = tf.shape(batch_nodes)[1]
            # ============ encode graph structure ==========
            fw_sampler = UniformNeighborSampler(fw_adj_info)
            bw_sampler = UniformNeighborSampler(bw_adj_info)
            nodes = tf.reshape(batch_nodes, [-1, ])

            # the fw_hidden and bw_hidden is the initial node embedding
            # [node_size, dim_size]
            fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
            bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)

            # [node_size, adj_size]
            fw_sampled_neighbors = fw_sampler((nodes, sample_size_per_layer))
            bw_sampled_neighbors = bw_sampler((nodes, sample_size_per_layer))

            inter_fw_hiddens = []
            inter_bw_hiddens = []
            inter_dims = []

            if scope == "first_gcn":
                self.watch["node_1_rep_in_first_gcn"] = []

            fw_hidden_dim = input_node_dim
            # layer is the index of convolution and hop is used to combine information
            for layer in range(layer_size):
                self.watch["node_1_rep_in_first_gcn"].append(fw_hidden)

                if len(fw_aggregators) <= layer:
                    fw_aggregators.append([])
                if len(bw_aggregators) <= layer:
                    bw_aggregators.append([])
                for hop in range(window_size):
                    if hop > 6:
                        fw_aggregator = fw_aggregators[layer][6]
                    elif len(fw_aggregators[layer]) > hop:
                        fw_aggregator = fw_aggregators[layer][hop]
                    else:
                        if agg_type == "GCN":
                            fw_aggregator = GCNAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                          dropout=self.dropout, mode=self.mode)
                        elif agg_type == "mean_pooling":
                            fw_aggregator = MeanAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                           dropout=self.dropout, if_use_high_way=self.with_gcn_highway, mode=self.mode)
                        elif agg_type == "max_pooling":
                            fw_aggregator = MaxPoolingAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                                 dropout=self.dropout, mode=self.mode)
                        elif agg_type == "lstm":
                            fw_aggregator = SeqAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                          dropout=self.dropout, mode=self.mode)
                        elif agg_type == "att":
                            fw_aggregator = AttentionAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                                dropout=self.dropout, mode=self.mode)

                        fw_aggregators[layer].append(fw_aggregator)

                    # [node_size, adj_size, word_embedding_dim]
                    if layer == 0 and hop == 0:
                        neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
                    else:
                        neigh_vec_hidden = tf.nn.embedding_lookup(
                            tf.concat([fw_hidden, tf.zeros([1, fw_hidden_dim])], 0), fw_sampled_neighbors)

                    # if self.with_gcn_highway:
                    #     # we try to forget something when introducing the neighbor information
                    #     with tf.variable_scope("fw_hidden_highway"):
                    #         fw_hidden = multi_highway_layer(fw_hidden, fw_hidden_dim, options['highway_layer_num'])

                    bw_hidden_dim = fw_hidden_dim

                    fw_hidden, fw_hidden_dim = fw_aggregator((fw_hidden, neigh_vec_hidden))

                    if keep_inter_state:
                        inter_fw_hiddens.append(fw_hidden)
                        inter_dims.append(fw_hidden_dim)

                    if self.graph_encode_direction == "bi":
                        if hop > 6:
                            bw_aggregator = bw_aggregators[layer][6]
                        elif len(bw_aggregators[layer]) > hop:
                            bw_aggregator = bw_aggregators[layer][hop]
                        else:
                            if agg_type == "GCN":
                                bw_aggregator = GCNAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                              dropout=self.dropout, mode=self.mode)
                            elif agg_type == "mean_pooling":
                                bw_aggregator = MeanAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                               dropout=self.dropout, if_use_high_way=self.with_gcn_highway, mode=self.mode)
                            elif agg_type == "max_pooling":
                                bw_aggregator = MaxPoolingAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                                     dropout=self.dropout, mode=self.mode)
                            elif agg_type == "lstm":
                                bw_aggregator = SeqAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                              dropout=self.dropout, mode=self.mode)
                            elif agg_type == "att":
                                bw_aggregator = AttentionAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                                    mode=self.mode, dropout=self.dropout)

                            bw_aggregators[layer].append(bw_aggregator)

                        if layer == 0 and hop == 0:
                            neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                        else:
                            neigh_vec_hidden = tf.nn.embedding_lookup(
                                tf.concat([bw_hidden, tf.zeros([1, fw_hidden_dim])], 0), bw_sampled_neighbors)

                        if self.with_gcn_highway:
                            with tf.variable_scope("bw_hidden_highway"):
                                bw_hidden = multi_highway_layer(bw_hidden, fw_hidden_dim, options['highway_layer_num'])

                        bw_hidden, bw_hidden_dim = bw_aggregator((bw_hidden, neigh_vec_hidden))

                        if keep_inter_state:
                            inter_bw_hiddens.append(bw_hidden)

            node_dim = fw_hidden_dim

            # hidden stores the representation for all nodes
            fw_hidden = tf.reshape(fw_hidden, [-1, single_graph_nodes_size, node_dim])
            if self.graph_encode_direction == "bi":
                bw_hidden = tf.reshape(bw_hidden, [-1, single_graph_nodes_size, node_dim])
                hidden = tf.concat([fw_hidden, bw_hidden], axis=2)
                graph_dim = 2 * node_dim
            else:
                hidden = fw_hidden
                graph_dim = node_dim

            hidden = tf.nn.relu(hidden)
            max_pooled = tf.reduce_max(hidden, 1)
            mean_pooled = tf.reduce_mean(hidden, 1)
            res = [hidden]

            max_graph_embedding = tf.reshape(max_pooled, [-1, graph_dim])
            mean_graph_embedding = tf.reshape(mean_pooled, [-1, graph_dim])
            res.append(max_graph_embedding)
            res.append(mean_graph_embedding)
            res.append(graph_dim)

            if keep_inter_state:
                inter_node_reps = []
                inter_graph_reps = []
                inter_graph_dims = []
                # process the inter hidden states
                for _ in range(len(inter_fw_hiddens)):
                    inter_fw_hidden = inter_fw_hiddens[_]
                    inter_bw_hidden = inter_bw_hiddens[_]
                    inter_dim = inter_dims[_]
                    inter_fw_hidden = tf.reshape(inter_fw_hidden, [-1, single_graph_nodes_size, inter_dim])

                    if self.graph_encode_direction == "bi":
                        inter_bw_hidden = tf.reshape(inter_bw_hidden, [-1, single_graph_nodes_size, inter_dim])
                        inter_hidden = tf.concat([inter_fw_hidden, inter_bw_hidden], axis=2)
                        inter_graph_dim = inter_dim * 2
                    else:
                        inter_hidden = inter_fw_hidden
                        inter_graph_dim = inter_dim

                    inter_node_rep = tf.nn.relu(inter_hidden)
                    inter_node_reps.append(inter_node_rep)
                    inter_graph_dims.append(inter_graph_dim)

                    max_pooled_tmp = tf.reduce_max(inter_node_rep, 1)
                    mean_pooled_tmp = tf.reduce_max(inter_node_rep, 1)
                    max_graph_embedding = tf.reshape(max_pooled_tmp, [-1, inter_graph_dim])
                    mean_graph_embedding = tf.reshape(mean_pooled_tmp, [-1, inter_graph_dim])
                    inter_graph_reps.append((max_graph_embedding, mean_graph_embedding))

                res.append(inter_node_reps)
                res.append(inter_graph_reps)
                res.append(inter_graph_dims)

            return res

    def act(self, sess, mode, dict, if_pred_on_dev):
        self.if_pred_on_dev = if_pred_on_dev

        feed_dict = {
            self.y_true : np.array(dict['y']),
            self.fw_adj_info_first : np.array(dict['fw_adj_info_first']),
            self.bw_adj_info_first : np.array(dict['bw_adj_info_first']),
            self.feature_info_first : np.array(dict['feature_info_first']),
            self.feature_len_first : np.array(dict['feature_len_first']),
            self.batch_nodes_first : np.array(dict['batch_nodes_first']),
            self.batch_mask_first : np.array(dict['batch_mask_first']),
            self.looking_table_first : np.array(dict['looking_table_first']),

            self.fw_adj_info_second : np.array(dict['fw_adj_info_second']),
            self.bw_adj_info_second : np.array(dict['bw_adj_info_second']),
            self.feature_info_second : np.array(dict['feature_info_second']),
            self.feature_len_second : np.array(dict['feature_len_second']),
            self.batch_nodes_second : np.array(dict['batch_nodes_second']),
            self.batch_mask_second : np.array(dict['batch_mask_second']),
            self.looking_table_second : np.array(dict['looking_table_second']),
        }

        if mode == "train" and not if_pred_on_dev:
            output_feeds = [self.watch, self.training_op, self.loss]
        elif mode == "test" or if_pred_on_dev:
            output_feeds = [self.y_pred]

        results = sess.run(output_feeds, feed_dict)
        return results