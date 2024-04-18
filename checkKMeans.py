import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

X = load_data("dataset")

# 初始化空列表，用于存储每个k值对应的轮廓系数
silhouette_scores = []

# 尝试不同的k值
for k in range(2, 10):
    # 创建并拟合KMeans模型
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# 绘制柱状图
plt.bar(range(2, 10), silhouette_scores)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Values of k')
plt.show()
