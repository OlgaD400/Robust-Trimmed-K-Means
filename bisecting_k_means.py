from sklearn.cluster import KMeans
import numpy as np


def bisecting_k_means(X: np.ndarray, k: int):
    current_clusters = 1
    split = 0
    fin_labels = np.zeros(X.shape[0])
    cluster_ind = np.arange(X.shape[0])

    while current_clusters != k:
        kmeans = KMeans(n_clusters=2).fit(X)
        current_clusters += 1
        split += 1
        cluster_centers = kmeans.cluster_centers_
        sse = [0] * 2

        fin_labels[cluster_ind] = kmeans.labels_ + 2 * split

        for point, label in zip(X, kmeans.labels_):
            sse[label] += np.square(point - cluster_centers[label]).sum()
        chosen_cluster = np.argmax(sse, axis=0)

        cluster_ind = np.where(fin_labels == (chosen_cluster + 2 * split))[0]

        chosen_cluster_data = X[kmeans.labels_ == chosen_cluster]
        X = chosen_cluster_data

    return fin_labels

