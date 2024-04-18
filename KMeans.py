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
def initialSelection(dataset, k):

    # 设置随机种子为固定值12
    np.random.seed(12)
    centroids_idx = np.random.choice(len(dataset), k, replace=False)
    prev_centroids = dataset[centroids_idx]
    return prev_centroids

# 初始簇中心
# centroids = initialSelection(dataset, 2)
# print(centroids)
# print("长度为 ", len(dataset[0]))
# print("长度为 ", len(centroids[0]))

def assignClusterIds(dataset, centroids):

    # 初始化一个空的距离矩阵，形状为 (数据点数量, 簇心数量),代表各数据点到各簇心的距离
    distances_matrix = np.zeros((dataset.shape[0], centroids.shape[0]))
    # 计算每个数据点到每个簇心的距离
    for i in range(dataset.shape[0]):  # 遍历每个数据点
        for j in range(centroids.shape[0]):  # 遍历每个簇心
            distances_matrix[i, j] = ComputeDistance(dataset[i], centroids[j])

    # 获取每行最小值所对应的列索引，即每个数据点所属的簇
    labels = np.argmin(distances_matrix, axis=1)

    return labels


def computeClusterRepresentatives(dataset, centroids, labels, k):

    for i in range(k):
        # Find points belonging to the i-th cluster
        cluster_points = dataset[labels == i]
        # Calculate the mean of the points in the cluster as the new centroid
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)

    return centroids


def KMeans(dataset, k, maxIter=10):

    labels = None
    prev_centroids = initialSelection(dataset, k)
    # print("初始簇心：")
    # print(prev_centroids)
    centroids = None

    for _ in range(maxIter):

        if centroids is None:
            centroids = initialSelection(dataset, k)

        # print("初始簇心：")
        # print(prev_centroids)
        labels = assignClusterIds(dataset, centroids)
        centroids = computeClusterRepresentatives(dataset, centroids, labels, k)
        # print("更新后簇心：")
        # print(centroids)

        if np.array_equal(prev_centroids, centroids):
            # print("最后初始簇心：")
            # print(prev_centroids)
            # print("最后更新簇心：")
            # print(centroids)
            break
        else:
            prev_centroids = centroids
        # prev_centroids = centroids

    return labels


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
                    a[i] = 0
            else:
                # compute b(obj)
                tempb = np.sum(distMatrix[i][cluster]) / clusterSize
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
clusteringSilhouette = {}
distMatrix = distanceMatrix(dataset)

for i in range(2, 10):
    labels = KMeans(dataset, i)
    clusters = create_clusters(labels)
    clusteringSilhouette[i] = silhouette(dataset, clusters, distMatrix)

plot_silhouttee(clusteringSilhouette)


# dataset = load_data("dataset")
# distMatrix = distanceMatrix(dataset)


# N = 10
# clust1 = np.random.normal(5, 2, (N,2))
# positive = np.random.multivariate_normal([0,0], np.eye(2), N)
# negative = np.random.multivariate_normal([2,2], np.eye(2), N)
