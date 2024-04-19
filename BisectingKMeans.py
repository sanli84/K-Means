import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# 读取数据集
def load_data(fname):
    try:
        features = []

        with open(fname) as F:
            next(F)  # skip the first line with feature names
            for line in F:
                p = line.strip().split(' ')
                features.append(np.array(p[1:], dtype=float))

        features = np.array(features)

        # 如果文件中只有一个数据点，则返回 None
        if len(features) < 2:
            return None

        return features

    except FileNotFoundError:
        print(f"Error: File '{fname}' not found.")
        return None
    except IOError:
        print(f"Error: Unable to read file '{fname}'.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# print(dataset)

# 计算两数据点之间的距离
def ComputeDistance(x, y):
    # Compute the Euclidean distance between x and y
    return np.linalg.norm(x - y)


# 初始随机选择数据集中的行索引为簇中心,
def initialSelection(cluster_to_split, k):

    # 设置随机种子为固定值12
    np.random.seed(12)
    centroids_idx = np.random.choice(len(cluster_to_split), k, replace=False)
    prev_centroids = []
    centroid1 = cluster_to_split[centroids_idx[0]]
    centroid2 = cluster_to_split[centroids_idx[1]]
    prev_centroids.append(centroid1)
    prev_centroids.append(centroid2)
    return prev_centroids

# 初始簇中心
# centroids = initialSelection(dataset, 2)
# print(centroids)
# print("长度为 ", len(dataset[0]))
# print("长度为 ", len(centroids[0]))


# 计算每个簇的簇内距离平方和
def computeClusterDistance(cluster, centroid):

    totalDistance = 0
    for datapoint in cluster:
        tempDistance = ComputeDistance(datapoint, centroid)
        totalDistance += tempDistance

    return totalDistance


# k是目标簇的数量
def assignClusterIds(dataset, centroids):

    # 初始化一个空的距离矩阵，形状为 (数据点数量, 簇心数量),代表各数据点到各簇心的距离
    dataset = np.array(dataset)
    centroids = np.array(centroids)
    distances_matrix = np.zeros((dataset.shape[0], centroids.shape[0]))
    # 计算每个数据点到每个簇心的距离
    for i in range(dataset.shape[0]):  # 遍历每个数据点
        for j in range(centroids.shape[0]):  # 遍历每个簇心
            distances_matrix[i, j] = ComputeDistance(dataset[i], centroids[j])

    # 获取每行最小值所对应的列索引，即每个数据点所属的簇
    labels = np.argmin(distances_matrix, axis=1)

    return labels


def computeCentroid(cluster):

    centroid = np.mean(cluster, axis=0)

    return centroid

#
# def KMeans(dataset, k, maxIter=10):
#
#     labels = None
#     prev_centroids = initialSelection(dataset, k)
#     # print("初始簇心：")
#     # print(prev_centroids)
#     centroids = None
#
#     for _ in range(maxIter):
#
#         if centroids is None:
#             centroids = initialSelection(dataset, k)
#
#         # print("初始簇心：")
#         # print(prev_centroids)
#         labels = assignClusterIds(dataset, centroids)
#         centroids = computeClusterRepresentatives(dataset, centroids, labels, k)
#         # print("更新后簇心：")
#         # print(centroids)
#
#         if np.array_equal(prev_centroids, centroids):
#             # print("最后初始簇心：")
#             # print(prev_centroids)
#             # print("最后更新簇心：")
#             # print(centroids)
#             break
#         else:
#             prev_centroids = centroids
#         # prev_centroids = centroids
#
#     return labels


def create_clusters(labels):
    clusters_dict = {}

    # 遍历 labels 数组
    for i, cluster_id in enumerate(labels):
        if cluster_id not in clusters_dict:
            # 如果当前簇不存在于字典中，则创建一个新的列表
            clusters_dict[cluster_id] = [i]
        else:
            # 如果当前簇已经存在于字典中，则将当前索引添加到相应的列表中
            clusters_dict[cluster_id].append(i)

    # 将字典的值转换成列表
    clusters = list(clusters_dict.values())

    return clusters

def BisectingKMeans(dataset, k):

    clusters = []
    # 初始化将整个数据集视为一个簇
    clusters.append(dataset)
    # print("初始化时clusters为：", clusters)

    # 初始化目标分裂簇
    cluster_to_split = clusters[0]
    # print("初始化时cluster_to_split为：", cluster_to_split)
    #初始化目标分裂簇的两个簇心
    centroids = initialSelection(cluster_to_split, 2)
    #获取根据初始两个簇心的两个聚类标签集
    labels = assignClusterIds(cluster_to_split, centroids)
    # print("初始化时labels为：", labels)
    #初始化两个子集
    subset1 = []
    subset2 = []

    # 遍历 labels 中的每个值，根据对应的索引划分 cluster_to_split
    for i, label in enumerate(labels):
        if label == 1:  # 假设1表示第一个子集，2表示第二个子集
            subset1.append(cluster_to_split[i])
        else:
            subset2.append(cluster_to_split[i])

    # print("分裂得到的subset1为：", subset1)
    # print("分裂得到的subset2为：", subset2)
    clusters.pop(0)
    clusters.append(subset1)
    clusters.append(subset2)
    # print("分裂后clusters为：", clusters)
    # print("clusters 长度为： ", len(clusters))

    while len(clusters) < k:
        sse_list = []

        for i, cluster in enumerate(clusters):
            total_distance = computeClusterDistance(cluster, centroids[i])
            # 将计算得到的总距离加入列表
            sse_list.append(total_distance)
        # 使用 argmax() 函数获取最大值的索引,也就是目标分裂簇的索引
        target_index = np.argmax(sse_list)
        cluster_to_split = clusters[target_index]


        # 获取根据初始两个簇心的两个聚类标签集
        labels = assignClusterIds(cluster_to_split, centroids)
        # print("初始化时labels为：", labels)
        # 初始化两个子集
        subset1 = []
        subset2 = []

        # 遍历 labels 中的每个值，根据对应的索引划分 cluster_to_split
        for i, label in enumerate(labels):
            if label == 1:  # 假设1表示第一个子集，2表示第二个子集
                subset1.append(cluster_to_split[i])
            else:
                subset2.append(cluster_to_split[i])

        # print("分裂得到的subset1为：", subset1)
        # print("分裂得到的subset2为：", subset2)
        centroids.pop(target_index)
        centroids.extend(initialSelection(cluster_to_split, 2))
        sse_list.pop(target_index)
        clusters.pop(target_index)
        clusters.append(subset1)
        clusters.append(subset2)
        # print(len(clusters))

    # 用列表推导式替换 clusters 中的数组为对应的 dataset 的行索引
    clusters_with_indices = [[np.where((dataset == point).all(axis=1))[0][0] for point in cluster] for cluster in
                             clusters]

    # 替换 clusters 中的数组为对应的 dataset 的行索引
    clusters = clusters_with_indices

    return clusters



# labels = KMeans(dataset, 5)
# clusters = create_clusters(labels)
# print(labels)
# print(clusters)

def distanceMatrix(dataset, dist=ComputeDistance):

    # Compute the number of objects in the dataset
    N = len(dataset)

    # Distance matrix
    distMatrix = np.zeros((N, N))
    # Compute pairwise distances between the objects
    for i in range(N):
        for j in range(N):
            # Distance is symmetric, so compute the distances between i and j only once
            if i < j:
                distMatrix[i][j] = dist(dataset[i], dataset[j])
                distMatrix[j][i] = distMatrix[i][j]

    return distMatrix

def silhouetteCoefficient(dataset, clusters, distMatrix):
    # Compute the number of objects in the dataset
    N = len(dataset)

    silhouette = [0 for i in range(N)]
    a = [0 for i in range(N)]
    b = [math.inf for i in range(N)]

    for (i, obj) in enumerate(dataset):
        for (cluster_id, cluster) in enumerate(clusters):
            clusterSize = len(cluster)
            if i in cluster:
                # compute a(obj)
                if clusterSize > 1:
                    a[i] = np.sum(distMatrix[i][cluster]) / (clusterSize - 1)

            else:
                # compute b(obj)
                if clusterSize > 0:
                    tempb = np.sum(distMatrix[i][cluster]) / clusterSize
                else:
                    tempb = np.inf
                if tempb < b[i]:
                    b[i] = tempb

    for i in range(N):
        silhouette[i] = 0 if a[i] == 0 else (b[i] - a[i]) / np.max([a[i], b[i]])

    return silhouette


def silhouette(dataset, clusters, distMatrix):
    return np.mean(silhouetteCoefficient(dataset, clusters, distMatrix))


def plot_silhouttee(clusteringSilhouette):
    plt.bar(range(len(clusteringSilhouette)), list(clusteringSilhouette.values()), align='center')
    plt.xticks(range(len(clusteringSilhouette)), list(clusteringSilhouette.keys()))
    plt.title("K-Means", fontsize=16)
    plt.xlabel('Number of clusters', fontsize=14)
    plt.ylabel('Silhouette coefficient', fontsize=14)
    plt.show()

dataset = load_data("dataset")
distMatrix = distanceMatrix(dataset)
clusteringSilhouette = {}

for i in range(1, 10):
    clusters = BisectingKMeans(dataset, i)
    clusteringSilhouette[i] = silhouette(dataset, clusters, distMatrix)
plot_silhouttee(clusteringSilhouette)
