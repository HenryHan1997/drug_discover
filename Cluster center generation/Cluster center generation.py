# -*- codeing = utf-8 -*-
# @Time: 2021-8-11 10:05
# @Author: Yourui Han
# @File: sort_PL_ware: PyCharm


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import matplotlib
import os
from collections import Counter  # 引入Counter


matplotlib.rcParams['font.size'] = 50
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def new_drug_produce(drug1, drug2, weight1, weight2):

    G1 = nx.MultiGraph()
    G1.add_nodes_from(drug1[0])
    G1.add_edges_from(drug1[1])

    G2 = nx.MultiGraph()
    G2.add_nodes_from(drug2[0])
    G2.add_edges_from(drug2[1])

    G1_1 = copy.deepcopy(G1)
    G2_1 = copy.deepcopy(G2)
    # 获取两个但drug可以断开的单键
    G1_change_edges = []
    for edge in G1.edges():
        if G1.number_of_edges(edge[0], edge[1]) == 1:
            if list(edge[0])[-1] != 'H' and list(edge[1])[-1] != 'H':
                G1_1.remove_edge(edge[0], edge[1])
                flag = nx.is_connected(G1_1)
                if flag == False:
                    G1_change_edges.append(edge)

    G2_change_edges = []
    for edge in G2.edges():
        if G2.number_of_edges(edge[0], edge[1]) == 1:
            if list(edge[0])[-1] != 'H' and list(edge[1])[-1] != 'H':
                G2_1.remove_edge(edge[0], edge[1])
                flag = nx.is_connected(G2_1)
                if flag == False:
                    G2_change_edges.append(edge)
    cut_edge_number = [len(G1_change_edges), len(G2_change_edges)]
    new_drug = []
    if len(G1_change_edges) != 0 and len(G2_change_edges) != 0:
        G1_change = copy.deepcopy(G1)
        G2_change = copy.deepcopy(G2)

        G1_martex = np.array(nx.to_numpy_matrix(G1_change))
        G2_martex = np.array(nx.to_numpy_matrix(G2_change))

        G1_random_edge = random.randint(0, len(G1_change_edges) - 1)
        G2_random_edge = random.randint(0, len(G2_change_edges) - 1)

        remove_edge1 = G1_change_edges[G1_random_edge]
        remove_edge2 = G2_change_edges[G2_random_edge]

        remove_node1 = remove_edge1[random.randint(0, 1)]
        remove_node2 = remove_edge2[random.randint(0, 1)]

        G1_change.remove_edge(remove_edge1[0], remove_edge1[1])
        G2_change.remove_edge(remove_edge2[0], remove_edge2[1])

        G1_subgraph = nx.connected_components(G1_change)
        G2_subgraph = nx.connected_components(G2_change)
        G1_subgraph_list = []
        for subgraph in G1_subgraph:
            G1_subgraph_list.append(nx.subgraph(G1_change, subgraph))
        for subgraph in G1_subgraph_list:
            if remove_node1 in subgraph.nodes():
                G1_subgraph_random_edges = subgraph.edges()
                G1_subgraph_random_nodes = subgraph.nodes()

        G2_subgraph_list = []
        for subgraph in G2_subgraph:
            G2_subgraph_list.append(nx.subgraph(G2_change, subgraph))
        for subgraph in G2_subgraph_list:
            if remove_node2 in subgraph.nodes():
                G2_subgraph_random_edges = subgraph.edges()
                G2_subgraph_random_nodes = subgraph.nodes()

        new_drug_edges = list(G1_subgraph_random_edges) + list(G2_subgraph_random_edges)
        new_drug_nodes = list(G1_subgraph_random_nodes) + list(G2_subgraph_random_nodes)

        new_drug_edges.append((remove_node1, remove_node2))

        new_drug = [new_drug_nodes, new_drug_edges]
        G_new = nx.MultiGraph()
        G_new.add_nodes_from(new_drug[0])
        G_new.add_edges_from(new_drug[1])
        G_new_martex = np.array(nx.to_numpy_matrix(G_new))
        best_fitness = weight1 * fitness(G1_martex, G_new_martex) + weight2 * fitness(G2_martex, G_new_martex)
        print(best_fitness)

        for i in range(10):
            print(i)
            G1_change = copy.deepcopy(G1)
            G2_change = copy.deepcopy(G2)

            G1_martex = np.array(nx.to_numpy_matrix(G1_change))
            G2_martex = np.array(nx.to_numpy_matrix(G2_change))

            print(remove_edge1, remove_edge2)

            remove_node1 = remove_edge1[random.randint(0, 1)]
            remove_node2 = remove_edge2[random.randint(0, 1)]

            G1_change.remove_edge(remove_edge1[0], remove_edge1[1])
            G2_change.remove_edge(remove_edge2[0], remove_edge2[1])

            G1_subgraph = nx.connected_components(G1_change)
            G2_subgraph = nx.connected_components(G2_change)
            G1_subgraph_list = []
            for subgraph in G1_subgraph:
                G1_subgraph_list.append(nx.subgraph(G1_change, subgraph))
            for subgraph in G1_subgraph_list:
                if remove_node1 in subgraph.nodes():
                    G1_subgraph_random_edges = subgraph.edges()
                    G1_subgraph_random_nodes = subgraph.nodes()

            G2_subgraph_list = []
            for subgraph in G2_subgraph:
                G2_subgraph_list.append(nx.subgraph(G2_change, subgraph))
            for subgraph in G2_subgraph_list:
                if remove_node2 in subgraph.nodes():
                    G2_subgraph_random_edges = subgraph.edges()
                    G2_subgraph_random_nodes = subgraph.nodes()

            tip_new_drug_edges = list(G1_subgraph_random_edges) + list(G2_subgraph_random_edges)

            tip_new_drug_nodes = list(G1_subgraph_random_nodes) + list(G2_subgraph_random_nodes)
            tip_new_drug_edges.append((remove_node1, remove_node2))
            tip_new_drug = [tip_new_drug_nodes, tip_new_drug_edges]

            G_new = nx.MultiGraph()
            G_new.add_nodes_from(tip_new_drug[0])
            G_new.add_edges_from(tip_new_drug[1])
            G_new_martex = np.array(nx.to_numpy_matrix(G_new))
            tip_fitness = weight1 * fitness(G1_martex, G_new_martex) + weight2 * fitness(G2_martex, G_new_martex)
            if tip_fitness < best_fitness:
                new_drug = tip_new_drug
                best_fitness = tip_fitness

    return new_drug, cut_edge_number


def fitness(martex1, martex2):
    # 计算适应度
    a = np.shape(martex1)
    a0 = a[0]
    b = np.shape(martex2)
    b0 = b[0]
    q = np.lcm(a0, b0)
    E = np.identity(int(q / a0))
    new_v1 = np.kron(martex1, E)
    I = np.identity(int(q / b0))
    new_v2 = np.kron(martex2, I)
    distance = np.linalg.norm(new_v1 - new_v2, 2)

    return distance

def sort_drug(all_graphs_list):

    drug1 = []
    drug2 = []
    best_fitness = 100000
    for i in range(len(all_graphs_list) - 1):
        for j in range(i + 1, len(all_graphs_list)):
            G1 = nx.MultiGraph()
            G1.add_nodes_from(all_graphs_list[i][0])
            G1.add_edges_from(all_graphs_list[i][1])

            G2 = nx.MultiGraph()
            G2.add_nodes_from(all_graphs_list[j][0])
            G2.add_edges_from(all_graphs_list[j][1])

            G1_martex = np.array(nx.to_numpy_matrix(G1))
            G2_martex = np.array(nx.to_numpy_matrix(G2))

            tip_fitness = fitness(G1_martex, G2_martex)
            if tip_fitness < best_fitness:
                best_fitness = tip_fitness
                drug1 = all_graphs_list[i]
                drug2 = all_graphs_list[j]

    return drug1, drug2

if __name__ == '__main__':
    old_all_graphs_list = []
    old_graphs_weight = []
    # 读取种子化合物并构造矩阵
    PL_seed_pdb_name = np.loadtxt('PL_second_drug_seed.txt', delimiter=" ", dtype=str)
    for i in range(len(PL_seed_pdb_name)):
        path = PL_seed_pdb_name[i] + '.pdb'
        with open(path) as pdbfile:
            count = 0
            pdb = []
            nodes_list = []
            edges_list = []
            for line in pdbfile:
                pdb.append(line.split())
                if line.split()[0] == 'HETATM':
                    nodes_list.append(PL_seed_pdb_name[i] + '_' + line.split()[1] + '_' + line.split()[-1])
            for line in pdb:
                if line[0] == 'CONECT':
                    for i in range(2, len(line)):
                        edge = (nodes_list[int(line[1]) - 1], nodes_list[int(line[i]) - 1])
                        edges_list.append(edge)
        old_all_graphs_list.append([nodes_list, edges_list])
        old_graphs_weight.append(1)

    all_new_drug_list = []
    # 随机进行选择重组种子
    for j in range(50):
        print('新药：', j)
        all_graphs_list = copy.deepcopy(old_all_graphs_list)
        graphs_weight = copy.deepcopy(old_graphs_weight)
        while len(all_graphs_list) >= 2:
            print('种子数目为：', len(all_graphs_list))
            # random_seeds = random.sample(range(0, len(all_graphs_list)), 2)
            drug1, drug2 = sort_drug(all_graphs_list)
            index1 = all_graphs_list.index(drug1)
            index2 = all_graphs_list.index(drug2)
            new_drug, cut_edge_number = new_drug_produce(drug1, drug2,
                                        graphs_weight[index1], graphs_weight[index2])
            if cut_edge_number[0] != 0 and cut_edge_number[1] != 0:
                all_graphs_list.remove(drug1)
                all_graphs_list.remove(drug2)

                remove_weight1 = graphs_weight[index1]
                remove_weight2 = graphs_weight[index2]
                graphs_weight.remove(remove_weight1)
                graphs_weight.remove(remove_weight2)

                all_graphs_list.append(new_drug)
                graphs_weight.append(remove_weight1 + remove_weight2)
            else:
                remove_weight1 = graphs_weight[index1]
                remove_weight2 = graphs_weight[index2]
                if cut_edge_number[0] == 0:
                    all_graphs_list.remove(drug1)
                    graphs_weight.remove(remove_weight1)
                if cut_edge_number[1] == 0:
                    all_graphs_list.remove(drug2)
                    graphs_weight.remove(remove_weight2)
        # G = nx.MultiGraph()
        # G.add_nodes_from(all_graphs_list[0][0])
        # G.add_edges_from(all_graphs_list[0][1])
        # nx.draw_networkx(G, with_labels=True, node_size=50)
        # print(all_graphs_list)
        # plt.show()

        all_new_drug_list.append((all_graphs_list[0][0], all_graphs_list[0][1]))
        print(all_new_drug_list)

    dic = {}
    for i in all_new_drug_list:
        dic[i] = all_new_drug_list.count(i)
    print(dic)
    for key in dic.keys():
        if dic[key] >= 2:
            print(dic[key])
    # b = dict(Counter(all_new_drug_list))
    # print([key for key, value in b.items() if value > 1])  # 只展示重复元素
    # print({key: value for key, value in b.items() if value > 1})  # 展现重复元素和重复次数