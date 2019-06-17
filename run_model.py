import os
import configure as conf
import data_collector as data_collector
import loaderAndwriter as disk_helper
import argparse
import tensorflow as tf
from model import GraphMatchNN
import numpy as np
from tqdm import tqdm
import datetime

def main():
    word_idx = {}
    model_type = conf.model_type
    epochs = conf.epochs
    pretrained_word_size = 0
    pretrained_word_embeddings = np.array([])
    if conf.if_use_pretrained_embedding:
        print("loading pretrained embedding ...")
        pretrained_word_embeddings = disk_helper.load_word_embedding(conf.pretrained_word_embedding_path, word_idx)
        pretrained_word_size = len(pretrained_word_embeddings)
        conf.hidden_layer_dim = conf.pretrained_word_embedding_dim

        print("load {} pre-trained word embeddings from Glove".format(pretrained_word_size))

    word_idx[conf.unknown_word] = len(word_idx.keys()) + 1

    conf.word_idx_file_path = "saved_model/" + conf.model_name + "/" + conf.word_idx_file_path
    conf.pred_file_path = "saved_model/" + conf.model_name + "/" + conf.pred_file_path

    if model_type == "train":

        np.random.seed(0)
        train_batch_size = conf.train_batch_size

        print("reading training data into the mem ...")

        graphs_1_train, graphs_2_train, labels_train = data_collector.read_data(conf.train_data_path, conf.graph_dir_name, word_idx, True)

        print("reading development data into the mem ...")
        graphs_1_dev, graphs_2_dev, labels_dev = data_collector.read_data(conf.dev_data_path, conf.graph_dir_name, word_idx, False)

        print("writing word-idx mapping ...")
        disk_helper.write_word_idx(word_idx, conf.word_idx_file_path)

        conf.word_vocab_size = len(word_idx)
        conf.pretrained_word_size = pretrained_word_size
        conf.learned_word_size = len(word_idx) - pretrained_word_size

        with tf.Graph().as_default():
            # tf.set_random_seed(0)
            with tf.Session() as sess:
                model = GraphMatchNN("train", conf, pretrained_word_embeddings)
                model._build_graph()

                saver = tf.train.Saver(max_to_keep=None)
                sess.run(tf.initialize_all_variables())

                def train_step(g1_v_batch, g2_v_batch, label_v_batch, if_pred_on_dev=False):
                    dict = {}
                    dict['fw_adj_info_first'] = g1_v_batch['g_fw_adj']
                    dict['bw_adj_info_first'] = g1_v_batch['g_bw_adj']
                    dict['feature_info_first'] = g1_v_batch['g_ids_features']
                    dict['feature_len_first'] = g1_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_first'] = g1_v_batch['g_nodes']
                    dict['batch_mask_first'] = g1_v_batch['g_mask']
                    dict['looking_table_first'] = g1_v_batch['g_looking_table']

                    dict['fw_adj_info_second'] = g2_v_batch['g_fw_adj']
                    dict['bw_adj_info_second'] = g2_v_batch['g_bw_adj']
                    dict['feature_info_second'] = g2_v_batch['g_ids_features']
                    dict['feature_len_second'] = g2_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_second'] = g2_v_batch['g_nodes']
                    dict['batch_mask_second'] = g2_v_batch['g_mask']
                    dict['looking_table_second'] = g2_v_batch['g_looking_table']

                    dict['y'] = label_v_batch

                    if not if_pred_on_dev:
                        watch, _, loss = model.act(sess, "train", dict, if_pred_on_dev)
                        return loss

                    else:
                        predicted = model.act(sess, "train", dict, if_pred_on_dev)
                        return predicted

                best_acc = 0.0
                for t in range(1, epochs + 1):
                    n_train = len(graphs_1_train)
                    temp_order = list(range(n_train))
                    np.random.shuffle(temp_order)

                    loss_sum = 0.0
                    for start in tqdm(range(0, n_train, train_batch_size)):
                        end = min(start + train_batch_size, n_train)
                        graphs_1 = []
                        graphs_2 = []
                        labels = []
                        for _ in range(start, end):
                            idx = temp_order[_]
                            graphs_1.append(graphs_1_train[idx])
                            graphs_2.append(graphs_2_train[idx])
                            labels.append(labels_train[idx])

                        batch_graph_1 = data_collector.batch_graph(graphs_1)
                        batch_graph_2 = data_collector.batch_graph(graphs_2)

                        g1_v_batch = data_collector.vectorize_batch_graph(batch_graph_1, word_idx)
                        g2_v_batch = data_collector.vectorize_batch_graph(batch_graph_2, word_idx)
                        label_v_batch = data_collector.vectorize_label(labels)

                        train_loss = train_step(g1_v_batch, g2_v_batch, label_v_batch, if_pred_on_dev=False)

                        loss_sum += train_loss

                    #####################  evaluate the model on the dev data #######################
                    print("evaluating the model on the dev data ...")
                    n_dev = len(graphs_1_dev)
                    dev_batch_size = conf.dev_batch_size
                    golds = []
                    predicted_res = []
                    g1_ori_ids = []
                    g2_ori_ids = []
                    for start in tqdm(range(0, n_dev, dev_batch_size)):
                        end = min(start + dev_batch_size, n_dev)
                        graphs_1 = []
                        graphs_2 = []
                        labels = []
                        for _ in range(start, end):
                            graphs_1.append(graphs_1_dev[_])
                            graphs_2.append(graphs_2_dev[_])
                            labels.append(labels_dev[_])
                            golds.append(labels_dev[_])

                            g1_ori_ids.append(graphs_1_dev[_]['g_id'])
                            g2_ori_ids.append(graphs_2_dev[_]['g_id'])

                        batch_graph_1 = data_collector.batch_graph(graphs_1)
                        batch_graph_2 = data_collector.batch_graph(graphs_2)

                        g1_v_batch = data_collector.vectorize_batch_graph(batch_graph_1, word_idx)
                        g2_v_batch = data_collector.vectorize_batch_graph(batch_graph_2, word_idx)
                        label_v_batch = data_collector.vectorize_label(labels)

                        predicted = train_step(g1_v_batch, g2_v_batch, label_v_batch, if_pred_on_dev=True)[0]

                        for _ in range(0, end - start):
                            predicted_res.append(predicted[_][1])  # add the prediction result into the bag

                    count = 0.0
                    correct_10 = 0.0
                    correct_1 = 0.0
                    cand_size = conf.dev_cand_size
                    assert len(predicted_res) % cand_size == 0
                    assert len(predicted_res) == len(g1_ori_ids)
                    assert len(g1_ori_ids) == len(g2_ori_ids)
                    number = int(len(predicted_res)/cand_size)
                    incorrect_pairs = []
                    for _ in range(number):
                        idx_score = {}
                        for idx in range(cand_size):
                            idx_score[ _ * cand_size + idx ] = predicted_res[ _ * cand_size + idx ]
                        idx_score_items = idx_score.items()
                        idx_score_items = sorted(idx_score_items, key=lambda d: d[1], reverse=True)

                        id_1 = g1_ori_ids[_ * cand_size]
                        id_2 = g2_ori_ids[_ * cand_size]

                        for sub_idx in range(min(10, len(idx_score_items))):
                            idx = idx_score_items[sub_idx][0]
                            if golds[idx] == 1:
                                correct_10 += 1.0
                                if sub_idx == 0:
                                    correct_1 += 1.0
                                else:
                                    incorrect_pairs.append((id_1, id_2))
                                break
                        count += 1.0

                    acc_10 = correct_10 / count
                    acc_1 = correct_1 / count

                    if acc_1 > best_acc:
                        best_acc = acc_1
                        save_path = "saved_model/" + conf.model_name + "/"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        path = saver.save(sess, save_path + 'model', global_step=0)
                        print("Already saved model to {}".format(path))

                        print('writing prediction file...')
                        with open(conf.pred_file_path, 'w') as f:
                            for (id_1, id_2) in incorrect_pairs:
                                f.write(str(id_1)+"\t"+str(id_2)+"\n")

                    time_str = datetime.datetime.now().isoformat()
                    print('-----------------------')
                    print('time:{}'.format(time_str))
                    print('Epoch', t)
                    print('Loss on train:{}'.format(loss_sum))
                    print('acc @1 on Dev:{}'.format(acc_1))
                    print('acc @10 on Dev:{}'.format(acc_10))
                    print('best acc @1 on Dev:{}'.format(best_acc))
                    print('-----------------------')

    if model_type == "test":

        print("reading word idx mapping from file ...")
        word_idx = disk_helper.read_word_idx_from_file(conf.word_idx_file_path)

        print("reading training data into the mem ...")
        graphs_1_test, graphs_2_test, labels_test = data_collector.read_data(conf.test_data_path, conf.graph_dir_name, word_idx, False)

        conf.word_vocab_size = len(word_idx)
        conf.pretrained_word_size = pretrained_word_size
        conf.learned_word_size = len(word_idx) - pretrained_word_size

        with tf.Graph().as_default():
            with tf.Session() as sess:
                model = GraphMatchNN("test", conf, pretrained_word_embeddings)
                model._build_graph()
                saver = tf.train.Saver(max_to_keep=None)

                model_path_name = "saved_model/" + conf.model_name + "/model-0"
                model_pred_path = "saved_model/" + conf.model_name + "/prediction.txt"

                saver.restore(sess, model_path_name)

                def test_step(g1_v_batch, g2_v_batch, label_v_batch):
                    dict = {}
                    dict['fw_adj_info_first'] = g1_v_batch['g_fw_adj']
                    dict['bw_adj_info_first'] = g1_v_batch['g_bw_adj']
                    dict['feature_info_first'] = g1_v_batch['g_ids_features']
                    dict['feature_len_first'] = g1_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_first'] = g1_v_batch['g_nodes']
                    dict['batch_mask_first'] = g1_v_batch['g_mask']
                    dict['looking_table_first'] = g1_v_batch['g_looking_table']
                    dict['entity_index_first'] = g1_v_batch['entity_index']

                    dict['fw_adj_info_second'] = g2_v_batch['g_fw_adj']
                    dict['bw_adj_info_second'] = g2_v_batch['g_bw_adj']
                    dict['feature_info_second'] = g2_v_batch['g_ids_features']
                    dict['feature_len_second'] = g2_v_batch['g_ids_feature_lens']
                    dict['batch_nodes_second'] = g2_v_batch['g_nodes']
                    dict['batch_mask_second'] = g2_v_batch['g_mask']
                    dict['looking_table_second'] = g2_v_batch['g_looking_table']
                    dict['entity_index_second'] = g2_v_batch['entity_index']

                    dict['y'] = label_v_batch
                    predicted = model.act(sess, "test", dict, if_pred_on_dev=False)
                    return predicted

                n_test = len(graphs_1_test)
                test_batch_size = conf.test_batch_size
                golds = []
                predicted_res = []
                g1_ori_ids = []
                g2_ori_ids = []
                for start in tqdm(range(0, n_test, test_batch_size)):
                    end = min(start + test_batch_size, n_test)
                    graphs_1 = []
                    graphs_2 = []
                    labels = []
                    for _ in range(start, end):
                        graphs_1.append(graphs_1_test[_])
                        graphs_2.append(graphs_2_test[_])
                        labels.append(labels_test[_])
                        golds.append(labels_test[_])

                        g1_ori_ids.append(graphs_1_test[_]['g_id'])
                        g2_ori_ids.append(graphs_2_test[_]['g_id'])

                    batch_graph_1 = data_collector.batch_graph(graphs_1)
                    batch_graph_2 = data_collector.batch_graph(graphs_2)

                    g1_v_batch = data_collector.vectorize_batch_graph(batch_graph_1, word_idx)
                    g2_v_batch = data_collector.vectorize_batch_graph(batch_graph_2, word_idx)
                    label_v_batch = data_collector.vectorize_label(labels)

                    predicted = test_step(g1_v_batch, g2_v_batch, label_v_batch)[0]

                    for _ in range(0, end - start):
                        predicted_res.append(predicted[_][1])  # add the prediction result into the bag

                count = 0.0
                correct_10 = 0.0
                correct_1 = 0.0
                cand_size = conf.test_cand_size
                assert len(predicted_res) % cand_size == 0
                assert len(predicted_res) == len(g1_ori_ids)
                assert len(g1_ori_ids) == len(g2_ori_ids)
                number = int(len(predicted_res) / cand_size)
                incorrect_pairs = []
                for _ in range(number):
                    idx_score = {}
                    for idx in range(cand_size):
                        idx_score[_ * cand_size + idx] = predicted_res[_ * cand_size + idx]
                    idx_score_items = idx_score.items()
                    idx_score_items = sorted(idx_score_items, key=lambda d: d[1], reverse=True)

                    id_1 = g1_ori_ids[_ * cand_size]
                    id_2 = g2_ori_ids[_ * cand_size]

                    for sub_idx in range(min(10, len(idx_score_items))):
                        idx = idx_score_items[sub_idx][0]
                        if golds[idx] == 1:
                            correct_10 += 1.0
                            if sub_idx == 0:
                                correct_1 += 1.0
                            else:
                                incorrect_pairs.append((id_1, id_2))
                            break
                    count += 1.0

                acc_10 = correct_10 / count
                acc_1 = correct_1 / count
                print('-----------------------')
                print('acc @1 on Test:{}'.format(acc_1))
                print('acc @10 on Test:{}'.format(acc_10))
                print('-----------------------')
                print('writing prediction file...')
                with open(conf.pred_file_path, 'w') as f:
                    for (id_1, id_2) in incorrect_pairs:
                        f.write(str(id_1) + "\t" + str(id_2) + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", type=str, choices=["train", "test"])
    argparser.add_argument("task", type=str, choices=["zh_en", "en_zh", "fr_en", "en_fr", "ja_en", "en_ja"])
    argparser.add_argument("name", type=str, help=("specify the name of the model"))
    argparser.add_argument("-gcn_window_size_first", type=int, default=conf.gcn_window_size_first, help="window size at first gcn")
    argparser.add_argument("-gcn_layer_size_first", type=int, default=conf.gcn_layer_size_first, help="layer size at first gcn")
    argparser.add_argument("-gcn_window_size_second", type=int, default=conf.gcn_window_size_second, help="window size at second gcn")
    argparser.add_argument("-gcn_layer_size_second", type=int, default=conf.gcn_layer_size_second, help="layer size at second gcn")
    argparser.add_argument("-aggregator_dim_first", type=int, default=conf.aggregator_dim_first, help="first gcn node rep dim")
    argparser.add_argument("-aggregator_dim_second", type=int, default=conf.aggregator_dim_second, help="second gcn node rep dim")
    argparser.add_argument("-gcn_type_first", type=str, default=conf.gcn_type_first, help = "first gcn type")
    argparser.add_argument("-gcn_type_second", type=str, default=conf.gcn_type_second, help = "second gcn type")
    argparser.add_argument("-sample_size_per_layer", type=int, default=conf.sample_size_per_layer, help="sample size per layer")
    argparser.add_argument("-epochs", type=int, default=conf.epochs, help="training epochs")
    argparser.add_argument("-learning_rate", type=float, default=conf.learning_rate, help="learning rate")
    argparser.add_argument("-hidden_layer_dim", type=int, default=conf.hidden_layer_dim)
    argparser.add_argument("-use_pretrained_embedding", action='store_true', default=False, help="if use glove embedding")
    argparser.add_argument("-with_match_highway", action='store_true', default=False, help="with match highway")
    argparser.add_argument("-cosine_MP_dim", type=int, default=conf.cosine_MP_dim, help="mp dim")
    argparser.add_argument("-drop_out", type=float, default=conf.dropout, help="dropout rate")
    argparser.add_argument("-pred_method", type=str, default=conf.pred_method, choices=['graph_level', 'node_level'])

    config = argparser.parse_args()

    conf.model_type = config.mode
    conf.gcn_window_size_first = config.gcn_window_size_first
    conf.gcn_window_size_second = config.gcn_window_size_second
    conf.sample_size_per_layer = config.sample_size_per_layer
    conf.epochs = config.epochs
    conf.learning_rate = config.learning_rate
    conf.hidden_layer_dim = config.hidden_layer_dim
    conf.aggregator_dim_first = config.aggregator_dim_first
    conf.aggregator_dim_second = config.aggregator_dim_second
    conf.gcn_layer_size_first = config.gcn_layer_size_first
    conf.gcn_layer_size_second = config.gcn_layer_size_second
    conf.gcn_type_first = config.gcn_type_first
    conf.gcn_type_second = config.gcn_type_second
    conf.cosine_MP_dim = config.cosine_MP_dim
    conf.dropout = config.drop_out
    conf.if_use_pretrained_embedding = config.use_pretrained_embedding
    conf.pred_method = config.pred_method
    conf.task = config.task

    conf.train_data_path = "DBP15K/" + conf.task + "/train.examples." + str(conf.train_cand_size)
    conf.dev_data_path = "DBP15K/" + conf.task + "/dev.examples." + str(conf.dev_cand_size)
    conf.test_data_path = "DBP15K/" + conf.task + "/test.examples." + str(conf.test_cand_size)
    conf.graph_dir_name = "DBP15K/" + conf.task + "/"


    if conf.if_use_pretrained_embedding:
        conf.hidden_layer_dim = conf.pretrained_word_embedding_dim

    conf.model_name = config.name + "_win1_" + str(conf.gcn_window_size_first) + "_win2_" + str(conf.gcn_window_size_second) + "_node1dim_" + str(conf.aggregator_dim_first) + "_node2dim_" + str(conf.aggregator_dim_second) \
                       + "_word_embedding_dim_" + str(conf.hidden_layer_dim) + "_layer1_" + str(conf.gcn_layer_size_first) + "_layer2_" + str(conf.gcn_layer_size_second) + "_first_gcn_type_" + conf.gcn_type_first + "_second_gcn_type_" + conf.gcn_type_second \
                       + "_cosine_MP_dim_" + str(conf.cosine_MP_dim) + "_drop_out_" + str(conf.dropout) + "_use_Glove_" + str(conf.if_use_pretrained_embedding) + "_pm_" + conf.pred_method + "_sample_size_per_layer_" + str(conf.sample_size_per_layer)
    main()
