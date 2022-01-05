# -*- coding = utf-8 -*-
# @Time : 2021/6/8 9:05
# @Author : YouRui Han
# @File : drug_clustering.py
# @Software : PyCharm

import numpy as np
import xlwt


class Hierarchical:
    def __init__(self):
        self.flag = []
        self.clusterset = []


def distance_matrix_creat(all_pro_Active_matrix):
    global distance_matrix
    distance_matrix = np.mat(np.zeros((len(all_pro_Active_matrix), len(all_pro_Active_matrix))))
    distance_matrix = np.array(distance_matrix)
    i = 0
    for key1 in all_pro_Active_matrix.keys():
        j = 0
        for key2 in all_pro_Active_matrix.keys():
            print(key1, key2)
            a = np.shape(all_pro_Active_matrix[key1])
            a0 = a[0]
            print('a0=', a0)
            b = np.shape(all_pro_Active_matrix[key2])
            b0 = b[0]
            print('b0=', b0)
            q = np.lcm(a0, b0)
            print('q=', q)
            E = np.identity(int(q / a0))
            new_v1 = np.kron(all_pro_Active_matrix[key1], E)
            I = np.identity(int(q / b0))
            new_v2 = np.kron(all_pro_Active_matrix[key2], I)
            distance = abs(np.linalg.norm(new_v1, 2) - np.linalg.norm(new_v2, 2))
            # distance = np.linalg.norm(new_v1 - new_v2, 2)
            # distance = np.linalg.norm(np.kron(all_protein_list[i], all_protein_list[j]),2)/
            #           (np.linalg.norm(all_protein_list[i],2)*np.linalg.norm(all_protein_list[j],2))
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
            j += 1
        i += 1

    return distance_matrix


def distance(v1, v2, distance_matrix, all_pro_Active_matrix):
    distance = 0
    # 平均距离
    for i in range(len(v1.flag)):
        for j in range(len(v2.flag)):
            key_list = list(all_pro_Active_matrix.keys())
            index1 = key_list.index(v1.flag[i])
            index2 = key_list.index(v2.flag[j])
            distance = distance + distance_matrix[index1][index2]
    distance = distance/(len(v1.clusterset) * len(v2.clusterset))
    # 最大距离
    # for i in range(len(v1.flag)):
    #     for j in range(len(v2.flag)):
    #         if distance_matrix[v1.flag[i]][v2.flag[j]] >= distance:
    #             distance = distance_matrix[v1.flag[i]][v2.flag[j]]
    return distance


def hcluster(all_pro_Active_matrix, n):
    if len(all_pro_Active_matrix) <= 0:
        print('invalid data')

    clusters = [Hierarchical() for i in range(len(all_pro_Active_matrix))]

    count = 0
    for key in all_pro_Active_matrix.keys():
        clusters[count].flag.append(key)
        clusters[count].clusterset.append(all_pro_Active_matrix[key])
        count += 1

    # 计算距离矩阵
    distance_matrix = distance_matrix_creat(all_pro_Active_matrix)
    workbook = xlwt.Workbook(encoding="utf-8")  # 创建workbook对象
    worksheet = workbook.add_sheet("sheet1")  # 创建工作表
    for i in range(len(all_pro_Active_matrix)):
        for j in range(len(all_pro_Active_matrix)):
            worksheet.write(i, j, distance_matrix[i][j])
    workbook.save("distance_matrix_PL_M_df2_no_normalize.xls")
    # 当簇的个数大于n时循环
    while len(clusters) > n:
        minDist = 100000000000000
        min_id1 = None
        min_id2 = None

        for m in range(len(clusters)-1):
            for k in range(m+1, len(clusters)):
                tmp_distance = distance(clusters[m], clusters[k], distance_matrix, all_pro_Active_matrix)
                if tmp_distance < minDist:
                    minDist = tmp_distance
                    min_id1 = m
                    min_id2 = k

        if min_id1 != None and min_id2 != None and minDist != 1000000000:
            new_flag = clusters[min_id1].flag + clusters[min_id2].flag
            new_clusterset = clusters[min_id1].clusterset + clusters[min_id2].clusterset
            newcluster = Hierarchical()
            newcluster.flag = new_flag
            newcluster.clusterset = new_clusterset
            del clusters[min_id2]
            del clusters[min_id1]
            clusters.append(newcluster)

    finalcluster = clusters
    return finalcluster


if __name__ == '__main__':
    # 构造原子字典，往对角线添加以区分原子
    atom_dictionary = {}
    atom_dictionary['H'] = 1
    atom_dictionary['C'] = 2
    atom_dictionary['O'] = 3
    atom_dictionary['N'] = 4
    atom_dictionary['BR'] = 5
    atom_dictionary['F'] = 6
    atom_dictionary['CL'] = 7
    atom_dictionary['S'] = 8
    atom_dictionary['I'] = 9
    atom_dictionary['O1-'] = 10
    atom_dictionary['N1+'] = 11

    # 读取Mpro化合物并构造矩阵
    Mpro_Active_pdb_name = np.loadtxt('Mpro_Active_name.txt', delimiter=" ", dtype=str)
    all_Mpro_Active_matrix = {}
    for i in range(len(Mpro_Active_pdb_name)):
        path = Mpro_Active_pdb_name[i] + '.pdb'
        with open(path) as pdbfile:
            count = 0
            pdb = []
            print(pdbfile)
            for line in pdbfile:
                pdb.append(line.split())
                if line.split()[0] == 'HETATM':
                    count = count + 1
            print(pdbfile)
            Mpro_matrix = np.zeros((count, count))
            for line in pdb:
                if line[0] == 'CONECT':
                    for atom in line:
                        if atom != 'CONECT':
                            if atom != line[1]:
                                Mpro_matrix[int(line[1])-1][int(atom)-1] += 1
                                Mpro_matrix[int(atom) - 1][int(line[1]) - 1] += 1
                if line[0] == 'HETATM':
                    Mpro_matrix[int(line[1]) - 1][int(line[1]) - 1] = atom_dictionary[line[-1]]
            # 归一化
            # for j in range(count):
            #     row_sum = sum(Mpro_matrix[j])
            #     for k in range(count):
            #         Mpro_matrix[j][k] = Mpro_matrix[j][k] / row_sum

            all_Mpro_Active_matrix[Mpro_Active_pdb_name[i] + '-M'] = Mpro_matrix

    # 读取PLpro化合物并构造矩阵
    PLpro_Active_pdb_name = np.loadtxt('PLpro_Active_name.txt', delimiter=" ", dtype=str)
    all_PLpro_Active_matrix = {}
    for i in range(len(PLpro_Active_pdb_name)):
        path = PLpro_Active_pdb_name[i] + '.pdb'
        with open(path) as pdbfile:
            count = 0
            pdb = []
            for line in pdbfile:
                pdb.append(line.split())
                if line.split()[0] == 'HETATM':
                    count = count + 1
            PLpro_matrix = np.zeros((count, count))
            for line in pdb:
                if line[0] == 'CONECT':
                    for atom in line:
                        if atom != 'CONECT':
                            if atom != line[1]:
                                PLpro_matrix[int(line[1]) - 1][int(atom) - 1] += 1
                                PLpro_matrix[int(atom) - 1][int(line[1]) - 1] += 1
                if line[0] == 'HETATM':
                    PLpro_matrix[int(line[1]) - 1][int(line[1]) - 1] = atom_dictionary[line[-1]]
            # 归一化
            # for j in range(count):
            #     row_sum = sum(PLpro_matrix[j])
            #     for k in range(count):
            #         PLpro_matrix[j][k] = PLpro_matrix[j][k] / row_sum

            all_PLpro_Active_matrix[PLpro_Active_pdb_name[i] + '-PL'] = PLpro_matrix

    all_pro_Active_matrix = {}
    all_pro_Active_matrix.update(all_Mpro_Active_matrix)
    all_pro_Active_matrix.update(all_PLpro_Active_matrix)
    print(list(all_pro_Active_matrix.keys()))

    # 层次聚类
    N = 2
    finalcluster = hcluster(all_pro_Active_matrix, N)
    print(finalcluster[0].flag, finalcluster[1].flag)

    # 存储矩阵
    # workbook = xlwt.Workbook(encoding="utf-8")  # 创建workbook对象
    # worksheet = workbook.add_sheet("sheet1")  # 创建工作表
    # worksheet.write(0, 0, 'Mpro_name')
    # worksheet.write(0, 1, 'Mpro_matrix')
    # Mpro_array_tmp = np.array(Mpro_matrix)
    # Mpro_array = str(Mpro_array_tmp.reshape((1, count*count))[0])
    # workbook.save("Mpro_matrix.xls")
