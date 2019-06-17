import numpy as np
import configure as conf
from tqdm import tqdm
import DBP15K.preprocessor as preprocessor

def read_data(input_path, graph_dir_name, word_idx, if_increase_dct):
    g1_triple_path = graph_dir_name + "triples_1"
    g1_feature_path = graph_dir_name + "id_features_1"
    g2_triple_path = graph_dir_name + "triples_2"
    g2_feature_path = graph_dir_name + "id_features_2"

    g1_map = preprocessor.gen_graph(g1_triple_path, g1_feature_path)
    g2_map = preprocessor.gen_graph(g2_triple_path, g2_feature_path)

    graphs_1 = []
    graphs_2 = []
    labels = []
    with open(input_path, 'r') as fr:
        lines = fr.readlines()
        for _ in tqdm(range(len(lines))):
            line = lines[_].strip()
            info = line.split("\t")
            id_1 = int(info[0])
            id_2 = int(info[1])
            label = int(info[2])

            graph_1 = g1_map[id_1]
            graph_2 = g2_map[id_2]

            graph_1['g_id'] = id_1
            graph_2['g_id'] = id_2

            graphs_1.append(graph_1)
            graphs_2.append(graph_2)
            labels.append(label)

            if if_increase_dct:
                features = [graph_1['g_ids_features'], graph_2['g_ids_features']]
                for f in features:
                    for id in f:
                        for w in f[id].split():
                            if w not in word_idx:
                                word_idx[w] = len(word_idx) + 1

    return graphs_1, graphs_2, labels



def vectorize_data(word_idx, texts):
    tv = []
    for text in texts:
        stv = []
        for w in text.split():
            if w not in word_idx:
                stv.append(word_idx[conf.unknown_word])
            else:
                stv.append(word_idx[w])
        tv.append(stv)
    return tv

def batch_graph(graphs):
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_nodes = []

    for g in graphs:
        id_adj = g['g_adj']
        features = g['g_ids_features']

        nodes = []

        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj

        id_gid_map = {}
        offset = len(g_ids_features.keys())
        for id in features.keys():
            id = int(id)
            g_ids_features[offset + id] = features[id]
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)

        for id in id_adj:
            adj = id_adj[id]
            id = int(id)
            g_id = id_gid_map[id]
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            for t in adj:
                t = int(t)
                g_t = id_gid_map[t]
                g_fw_adj[g_id].append(g_t)
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                g_bw_adj[g_t].append(g_id)

    node_size = len(g_ids_features.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    graph = {}
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj'] = g_fw_adj
    graph['g_bw_adj'] = g_bw_adj

    return graph

def vectorize_label(labels):
    lv = []
    for label in labels:
        if label == 0 or label == '0':
            lv.append([1, 0])
        elif label == 1 or label == '1':
            lv.append([0, 1])
        else:
            print("error in vectoring the label")
    lv = np.array(lv)
    return lv


def vectorize_batch_graph(graph, word_idx):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}
    nv = []
    n_len_v = []
    word_max_len = 0
    for id in id_features:
        feature = id_features[id]
        word_max_len = max(word_max_len, len(feature.split()))
    # word_max_len = min(word_max_len, conf.word_size_max)

    for id in graph['g_ids_features']:
        feature = graph['g_ids_features'][id]
        fv = []
        for token in feature.split():
            if len(token) == 0:
                continue
            if token in word_idx:
                fv.append(word_idx[token])
            else:
                fv.append(word_idx[conf.unknown_word])

        if len(fv) > word_max_len:
            n_len_v.append(word_max_len)
        else:
            n_len_v.append(len(fv))

        for _ in range(word_max_len - len(fv)):
            fv.append(0)
        fv = fv[:word_max_len]
        nv.append(fv)

    # add an all-zero vector for the PAD node
    nv.append([0 for temp in range(word_max_len)])
    n_len_v.append(0)

    gv['g_ids_features'] = np.array(nv)
    gv['g_ids_feature_lens'] = np.array(n_len_v)

    # ============== vectorize adj info ======================
    g_fw_adj = graph['g_fw_adj']
    g_fw_adj_v = []

    degree_max_size = 0
    for id in g_fw_adj:
        degree_max_size = max(degree_max_size, len(g_fw_adj[id]))
    g_bw_adj = graph['g_bw_adj']
    for id in g_bw_adj:
        degree_max_size = max(degree_max_size, len(g_bw_adj[id]))
    degree_max_size = min(degree_max_size, conf.sample_size_per_layer)

    for id in g_fw_adj:
        adj = g_fw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_fw_adj.keys()))
        adj = adj[:degree_max_size]
        assert len(adj) == degree_max_size
        g_fw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

    g_bw_adj_v = []
    for id in g_bw_adj:
        adj = g_bw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_bw_adj.keys()))
        adj = adj[:degree_max_size]
        assert len(adj) == degree_max_size
        g_bw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

    # ============== vectorize nodes info ====================
    g_nodes = graph['g_nodes']
    graph_max_size = 0
    for nodes in g_nodes:
        graph_max_size = max(graph_max_size, len(nodes))

    g_node_v = []
    g_node_mask = []
    entity_index = []
    for nodes in g_nodes:
        mask = [1 for _ in range(len(nodes))]
        for _ in range(graph_max_size - len(nodes)):
            nodes.append(len(g_fw_adj.keys()))
            mask.append(0)
        nodes = nodes[:graph_max_size]
        mask = mask[:graph_max_size]
        g_node_v.append(nodes)
        g_node_mask.append(mask)
        entity_index.append(0)

    g_looking_table = []
    global_count = 0
    for mask in g_node_mask:
        for item in mask:
            if item == 1:
                g_looking_table.append(global_count)
            global_count += 1

    gv['g_nodes'] =np.array(g_node_v)
    gv['g_bw_adj'] = np.array(g_bw_adj_v)
    gv['g_fw_adj'] = np.array(g_fw_adj_v)
    gv['g_mask'] = np.array(g_node_mask)
    gv['g_looking_table'] = np.array(g_looking_table)
    gv['entity_index'] = entity_index

    return gv
