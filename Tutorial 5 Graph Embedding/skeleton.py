import ujson as json
import node2vec
import networkx as nx
from gensim.models import Word2Vec
import logging
import random
import numpy as np
from sklearn.metrics import roc_auc_score


def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]


def get_G_from_edges(edges):
    edge_dict = dict()
    # calculate the count for all the edges
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        # add edges to the graph
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        # add weights for all the edges
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G


def randomly_choose_false_edges(nodes, true_edges):
    tmp_list = list()
    all_edges = list()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            all_edges.append((i, j))
    random.shuffle(all_edges)
    for edge in all_edges:
        if edge[0] == edge[1]:
            continue
        if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (nodes[edge[1]], nodes[edge[0]]) not in true_edges:
            tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
    return tmp_list


def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return random.random()


def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


directed = True
p = 1
q = 1
num_walks = 10
walk_length = 10
dimension = 200
window_size = 5
num_workers = 4
iterations = 10
number_of_groups = 5

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Start to load the raw network

raw_edges = list()
with open('Dataset/CKM-Physicians-Innovation_multiplex.edges', 'r') as f:
    for line in f:
        head = line[:-1].split(' ')[1]
        tail = line[:-1].split(' ')[2]
        raw_edges.append((head, tail))
raw_edges = list(set(raw_edges))
edges_by_group = divide_data(raw_edges, number_of_groups)

All_AUC_scores = list()
for i in range(number_of_groups):
    print('We are working on group:', i)
    train_edges = list()
    test_edges = list()
    for j in range(number_of_groups):
        if i != j:
            train_edges += edges_by_group[j]
        else:
            test_edges += edges_by_group[j]

    train_nodes = list()
    for e in train_edges:
        train_nodes.append(e[0])
        train_nodes.append(e[1])
    train_nodes = list(set(train_nodes))

    negative_edges = randomly_choose_false_edges(train_nodes, train_edges + test_edges)[:len(test_edges)]

    # your code here
    # Create a node2vec object with training edges
    G = node2vec.Graph(get_G_from_edges(train_edges), directed, p, q)
    # Calculate the probability for the random walk process
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    model = Word2Vec(walks, size=dimension, window=window_size, min_count=0, sg=1, workers=num_workers, iter=iterations)
	
    resulted_embeddings = dict()
    for i, w in enumerate(model.wv.index2word):
        resulted_embeddings[w] = model.wv.syn0[i]

    tmp_AUC_score = get_AUC(model, test_edges, negative_edges)
    All_AUC_scores.append(tmp_AUC_score)
    print('tmp_accuracy:', tmp_AUC_score)

All_AUC_scores = np.asarray(All_AUC_scores)
print('Average AUC score:', np.average(All_AUC_scores))
print('std:', np.std(All_AUC_scores))
print('end')
