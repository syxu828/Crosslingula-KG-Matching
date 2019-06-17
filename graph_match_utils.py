import tensorflow as tf
import layer_utils

eps = 1e-6

def cosine_distance(y1, y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(node_1_repres, node_2_repres, watch=None):
    # [batch_size, 1, single_graph_1_nodes_size, node_embedding_dim]
    node_1_repres_tmp = tf.expand_dims(node_1_repres, 1)

    # [batch_size, single_graph_2_nodes_size, 1, node_embedding_dim]
    node_2_repres_tmp = tf.expand_dims(node_2_repres, 2)

    # [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    relevancy_matrix = cosine_distance(node_1_repres_tmp, node_2_repres_tmp)

    watch["node_1_repres_tmp"] = node_1_repres
    watch["node_2_repres_tmp"] = node_2_repres
    watch["relevancy_matrix"] = relevancy_matrix

    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, graph_1_mask, graph_2_mask):
    # relevancy_matrix: [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    # graph_1_mask: [batch_size, single_graph_1_nodes_size]
    # graph_2_mask: [batch_size, single_graph_2_nodes_size]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(graph_1_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(graph_2_mask, 2))

    # [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    return relevancy_matrix

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    # [batch_size, 'x', dim]
    in_tensor = tf.expand_dims(in_tensor, axis=1)
    # [1, decompse_dim, dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0)
    # [batch_size, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)

def cal_maxpooling_matching(node_1_rep, node_2_rep, decompose_params):
    # node_1_rep: [batch_size, single_graph_1_nodes_size, dim]
    # node_2_rep: [batch_size, single_graph_2_nodes_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        # p: [single_graph_1_nodes_size, dim], q: [single_graph_2_nodes_size, dim]
        p = x[0]
        q = x[1]

        # [single_graph_1_nodes_size, decompose_dim, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params)

        # [single_graph_2_nodes_size, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params)

        # [single_graph_1_nodes_size, 1, decompose_dim, dim]
        p = tf.expand_dims(p, 1)

        # [1, single_graph_2_nodes_size, decompose_dim, dim]
        q = tf.expand_dims(q, 0)

        # [single_graph_1_nodes_size, single_graph_2_nodes_size, decompose]
        return cosine_distance(p, q)

    elems = (node_1_rep, node_2_rep)

    # [batch_size, single_graph_1_nodes_size, single_graph_2_nodes_size, decompse_dim]
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32)

    # [batch_size, single_graph_1_nodes_size, 2 * decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])

def cal_max_node_2_representation(node_2_rep, relevancy_matrix):
    # [batch_size, single_graph_1_nodes_size]
    atten_positions = tf.argmax(relevancy_matrix, axis=2, output_type=tf.int32)
    max_node_2_reps = layer_utils.collect_representation(node_2_rep, atten_positions)

    # [batch_size, single_graph_1_nodes_size, dim]
    return max_node_2_reps

def multi_perspective_match(feature_dim, rep_1, rep_2, options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    input_shape = tf.shape(rep_1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if options['with_cosine']:
            cosine_value = layer_utils.cosine_distance(rep_1, rep_2, cosine_norm=False)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
            match_dim += 1

        if options['with_mp_cosine']:
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[options['cosine_MP_dim'], feature_dim],
                                               dtype=tf.float32)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            rep_1_flat = tf.expand_dims(rep_1, axis=2)
            rep_2_flat = tf.expand_dims(rep_2, axis=2)
            mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(rep_1_flat, mp_cosine_params),
                                                             rep_2_flat, cosine_norm=False)
            matching_result.append(mp_cosine_matching)
            match_dim += options['cosine_MP_dim']

    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)

def match_graph_1_with_graph_2(node_1_rep, node_2_rep, node_1_mask, node_2_mask, node_rep_dim, options=None, watch=None):
    '''

    :param node_1_rep:
    :param node_2_rep:
    :param node_1_mask:
    :param node_2_mask:
    :param node_rep_dim: dim of node representation
    :param with_maxpool_match:
    :param with_max_attentive_match:
    :param options:
    :return:
    '''

    with_maxpool_match = options["with_maxpool_match"]
    with_max_attentive_match = options["with_max_attentive_match"]

    # an array of [batch_size, single_graph_1_nodes_size]
    all_graph_2_aware_representations = []
    dim = 0
    with tf.variable_scope('match_graph_1_with_graph_2'):
        # [batch_size, single_graph_1_nodes_size, single_graph_2_nodes_size]
        relevancy_matrix = cal_relevancy_matrix(node_2_rep, node_1_rep, watch=watch)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, node_2_mask, node_1_mask)

        all_graph_2_aware_representations.append(tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
        all_graph_2_aware_representations.append(tf.reduce_mean(relevancy_matrix, axis=2, keep_dims=True))
        dim += 2

        if with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                       shape=[options['cosine_MP_dim'], node_rep_dim],
                                                       dtype=tf.float32)

            # [batch_size, single_graph_1_nodes_size, 2 * decompse_dim]
            maxpooling_rep = cal_maxpooling_matching(node_1_rep, node_2_rep, maxpooling_decomp_params)
            maxpooling_rep = tf.multiply(maxpooling_rep, tf.expand_dims(node_1_mask, -1))
            all_graph_2_aware_representations.append(maxpooling_rep)
            dim += 2 * options['cosine_MP_dim']

        if with_max_attentive_match:
            # [batch_size, single_graph_1_nodes_size, dim]
            max_att = cal_max_node_2_representation(node_2_rep, relevancy_matrix)

            # [batch_size, single_graph_1_nodes_size, match_dim]
            (max_attentive_rep, match_dim) = multi_perspective_match(node_rep_dim, node_1_rep, max_att, options=options, scope_name='mp-match-max-att')
            max_attentive_rep = tf.multiply(max_attentive_rep, tf.expand_dims(node_1_mask, -1))
            all_graph_2_aware_representations.append(max_attentive_rep)
            dim += match_dim

        # [batch_size, single_graph_1_nodes_size, dim]
        all_graph_2_aware_representations = tf.concat(axis=2, values=all_graph_2_aware_representations)

    return (all_graph_2_aware_representations, dim)