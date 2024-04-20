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


# Initially randomly select the row index in the data set as the cluster centroids
# input: dataset, number of clusters
# output: centroids
def initialSelection(dataset, k):

    # Set the random seed to a fixed value of 12
    np.random.seed(12)
    centroids_idx = np.random.choice(len(dataset), k, replace=False)  # choose random row index from dataset
    prev_centroids = dataset[centroids_idx]  # get the datapoint of the index
    return prev_centroids


# function of assigning cluster ID to each datapoints
# input: dataset, current centroids
# output: arrays of labels for all datapoints
def assignClusterIds(dataset, centroids):

    # Initialize an empty distance matrix with the shape of (number of data points, number of centroids),
    # representing the distance from each data point to each centroid.
    distances_matrix = np.zeros((dataset.shape[0], centroids.shape[0]))
    # Calculate the distance from each data point to each centroid
    for i in range(dataset.shape[0]):  # Loop through each data point
        for j in range(centroids.shape[0]):  # Loop through each centroids
            distances_matrix[i, j] = ComputeDistance(dataset[i], centroids[j])

    # Get the column index corresponding to the minimum value of each row,
    # that is, the cluster to which each data point belongs
    labels = np.argmin(distances_matrix, axis=1)

    return labels


# function of computing the representatives for each cluster
# input: dataset matrix, centroids array, labels array, number of clusters
# output: updated centroids for each cluster
def computeClusterRepresentatives(dataset, centroids, labels, k):

    for i in range(k):
        # Find points belonging to the i-th cluster
        cluster_points = dataset[labels == i]
        # Calculate the mean of the points in the cluster as the new centroid
        # The average value of each point in the cluster is the centroid
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)

    return centroids


# KMeans clustering function
# input: dataset matrix, number of target clusters, max iteration times
# output: labels array for all datapoints
def KMeans(dataset, k, maxIter=10):

    if k ==1:
        return np.ones(len(dataset))

    labels = None
    # get centroids in last iteration
    prev_centroids = initialSelection(dataset, k)
    centroids = None

    for _ in range(maxIter):

        if centroids is None:
            # if there is no centroid in the first iteration, choose one randomly
            centroids = initialSelection(dataset, k)

        # assign cluster ID to each datapoint
        labels = assignClusterIds(dataset, centroids)
        # computing centroids according to updated labels
        centroids = computeClusterRepresentatives(dataset, centroids, labels, k)

        if np.array_equal(prev_centroids, centroids):
            break
        else:
            # updated centroids for next iteration comparison
            prev_centroids = centroids

    return labels

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
                    a[i] = 0
            else:
                # compute b(obj)
                tempb = np.sum(distMatrix[i][cluster]) / clusterSize
                if tempb < b[i]:
                    b[i] = tempb

    for i in range(N):
        if np.isnan(a[i]) or np.isnan(b[i]) or np.isinf(a[i]) or np.isinf(b[i]):
            silhouette[i] = 0
        elif (a[i] == 0) or (b[i] == 0):
            silhouette[i] = 0
        else:
            silhouette[i] = (b[i] - a[i]) / np.max([a[i], b[i]])

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
clusteringSilhouette = {}
distMatrix = distanceMatrix(dataset)

# loop through from 1 cluster to 9 clusters
for i in range(1, 10):
    labels = KMeans(dataset, i)  # get labels for the data points
    clusters = create_clusters(labels)  # convert labels to clusters
    clusteringSilhouette[i] = silhouette(dataset, clusters, distMatrix)

plot_silhouttee(clusteringSilhouette)
plt.savefig('KMeans.png')
