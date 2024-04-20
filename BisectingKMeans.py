import math

import numpy as np
import matplotlib.pyplot as plt


# read dataset function
# input: dataset file
# output: feature matrix
def load_data(fname):
    try:
        features = []  # create feature matrix

        with open(fname) as F:
            next(F)  # skip the first line with feature names
            for line in F:
                p = line.strip().split(' ')
                features.append(np.array(p[1:], dtype=float))

        features = np.array(features)

        # If the number of datapoints in the file is less than 2, return None
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


# function of computing distance between two data points
# input: two data points
# output: float distance between two points
def ComputeDistance(x, y):
    # Compute the Euclidean distance between x and y
    return np.linalg.norm(x - y)


# Initial random selection of row index in target cluster as centroid
# input: target cluster, number of clusters(2 because of Bisecting)
# output: 2 centroids in target cluster
def initialSelection(cluster_to_split, k):

    # Set the random seed to a fixed value of 12
    np.random.seed(12)
    # get 2 indexes of centroids in target cluster
    centroids_idx = np.random.choice(len(cluster_to_split), k, replace=False)
    prev_centroids = []  # centroids array
    # get centroids from target cluster according to index
    centroid1 = cluster_to_split[centroids_idx[0]]
    centroid2 = cluster_to_split[centroids_idx[1]]
    # store two centroids into centroids array
    prev_centroids.append(centroid1)
    prev_centroids.append(centroid2)
    return prev_centroids


#  function of calculating the sum of squared intra-cluster distances for each cluster
# input: cluster, centroid of the cluster
# output: the sum of square Euclidean distance in the cluster
def computeSumSquare(cluster, centroid):

    totalDistance = 0
    for datapoint in cluster:
        tempDistance = ComputeDistance(datapoint, centroid)
        totalDistance += tempDistance

    return totalDistance


# function of assigning cluster ID to each datapoints
# input: dataset, current centroids
# output: arrays of labels for all datapoints
def assignClusterIds(dataset, centroids):

    # Initialize an empty distance matrix with the shape of (number of data points, number of centroids),
    # representing the distance from each data point to each centroid.
    dataset = np.array(dataset)
    centroids = np.array(centroids)
    distances_matrix = np.zeros((dataset.shape[0], centroids.shape[0]))
    # Calculate the distance from each data point to each centroid
    for i in range(dataset.shape[0]):  # Loop through each data point
        for j in range(centroids.shape[0]):  # Loop through each centroids
            distances_matrix[i, j] = ComputeDistance(dataset[i], centroids[j])

    # Get the column index corresponding to the minimum value of each row,
    # that is, the cluster to which each data point belongs
    labels = np.argmin(distances_matrix, axis=1)

    return labels


# function of computing centroids in target cluster
# input: array of all datapoints in one cluster
# output: centroid of target cluster
def computeCentroid(cluster):
    # computing mean value of each cluster
    centroid = np.mean(cluster, axis=0)
    return centroid



# function of converting labels array to cluster array
# input: labels array
# output: cluster array
def create_clusters(labels):
    clusters_dict = {}

    # loop through labels array
    for i, cluster_id in enumerate(labels):
        if cluster_id not in clusters_dict:
            # If the current cluster does not exist in the dictionary, create a new list
            clusters_dict[cluster_id] = [i]
        else:
            # If the current cluster already exists in the dictionary, add the current index to the corresponding list
            clusters_dict[cluster_id].append(i)

    # Convert dictionary values into lists
    clusters = list(clusters_dict.values())

    return clusters


# function of Bisecting K-Means clustering
# input: dataset matrix, number of clusters
# output: labels array for all datapoints
def BisectingKMeans(dataset, k):

    clusters = []
    # Initialize the entire data set as a cluster
    clusters.append(dataset)

    # Initialize the target split cluster
    cluster_to_split = clusters[0]
    # Initialize the two cluster centers of the target split cluster
    centroids = initialSelection(cluster_to_split, 2)
    # Get two cluster label sets based on the initial two cluster centers
    labels = assignClusterIds(cluster_to_split, centroids)
    # Initialize two subsets
    subset1 = []
    subset2 = []

    # Traverse each value in labels and divide cluster_to_split according to the corresponding index
    for i, label in enumerate(labels):
        if label == 1:  # 1 represents the first subset and 2 represents the second subset
            subset1.append(cluster_to_split[i])
        else:
            subset2.append(cluster_to_split[i])

    clusters.pop(0) # pop the previous undivided cluster
    clusters.append(subset1) # append the two divided cluster
    clusters.append(subset2)

    while len(clusters) < k:
        sse_list = []

        for i, cluster in enumerate(clusters):
            total_distance = computeSumSquare(cluster, centroids[i])
            # Add the calculated total distance to the list
            sse_list.append(total_distance)
        # Use the argmax() function to obtain the index of the maximum value,
        # which is the index of the target split cluster
        target_index = np.argmax(sse_list)
        cluster_to_split = clusters[target_index]


        # Get two cluster label sets based on the initial two centroids
        labels = assignClusterIds(cluster_to_split, centroids)
        # Initialize two subsets
        subset1 = []
        subset2 = []

        # Traverse each value in labels and divide cluster_to_split according to the corresponding index
        for i, label in enumerate(labels):
            if label == 1:  # 1 represents the first cluster, 2 represents the second cluster
                subset1.append(cluster_to_split[i])
            else:
                subset2.append(cluster_to_split[i])

        centroids.pop(target_index)
        centroids.extend(initialSelection(cluster_to_split, 2))
        sse_list.pop(target_index)
        # pop up the previous undivided target cluster
        clusters.pop(target_index)
        # append two divided cluster into clusters
        clusters.append(subset1)
        clusters.append(subset2)

    # Use list comprehensions to replace the arrays in clusters with the row index of the corresponding dataset
    clusters_with_indices = [[np.where((dataset == point).all(axis=1))[0][0] for point in cluster] for cluster in
                             clusters]

    # Replace the array in clusters with the row index of the corresponding dataset
    clusters = clusters_with_indices

    return clusters


# function of creating the distance matrix of all datapoints in the dataset
# input: dataset matrix, distance computing parameters
# output: distance matrix
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


# function of computing silhouette Coefficient
# reuse of week 7 silhouetteCoefficient function
# input: dataset matrix, clusters array, distance matrix
def silhouetteCoefficient(dataset, clusters, distMatrix):
    # Compute the number of objects in the dataset
    N = len(dataset)

    # according to the definition of silhouette Coefficient,
    # when cluster number equals to 1, silhouette Coefficient equals to 0
    if N == 1:
        return 0

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


# reuse of silhouette function in week 7
# input: dataset matrix, clusters array, distance matrix
# output: silhouette Coefficient score
def silhouette(dataset, clusters, distMatrix):
    return np.mean(silhouetteCoefficient(dataset, clusters, distMatrix))


# function of plotting bar chart of cluster number VS silhouette Coefficient
# input: silhouette Coefficient array
# output: barchart of cluster number VS silhouette Coefficient
def plot_silhouttee(clusteringSilhouette):
    plt.bar(range(len(clusteringSilhouette)), list(clusteringSilhouette.values()), align='center')
    plt.xticks(range(len(clusteringSilhouette)), list(clusteringSilhouette.keys()))
    plt.title("K-Means", fontsize=16)
    plt.xlabel('Number of clusters', fontsize=14)
    plt.ylabel('Silhouette coefficient', fontsize=14)
    plt.show()

# read dataset and convert it to matrix
dataset = load_data("dataset")
distMatrix = distanceMatrix(dataset)
clusteringSilhouette = {}


# loop through from 1 cluster to 9 clusters
for i in range(1, 10):
    clusters = BisectingKMeans(dataset, i)  # get labels for the data points
    clusteringSilhouette[i] = silhouette(dataset, clusters, distMatrix)
plot_silhouttee(clusteringSilhouette)
plt.savefig('BisectingKMeans.png')