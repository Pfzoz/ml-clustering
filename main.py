
# coesao, separacao e coef de silhueta -> interno
# homegeneidade, completude, entropia, indice randomico -> externo
#
#
# aglomerativa, dbscan e kmeans
#
#
# kmeans: n clusters e max iter
# dbscan: eps e min samples
# agnes: n clusters e linkage
#
# análise descritiva
#

import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances, homogeneity_score, completeness_score, adjusted_rand_score
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder, normalize

# Load the data

data = pd.read_csv("new_data.csv")
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

print(data.head())

data.drop(columns=["run_ID", "rerun_ID", "obj_ID", "fiber_ID", "cam_col", "field_ID", "spec_obj_ID"])

Y = data['class']

data = data.drop(columns=["class"])

X = data
X = normalize(X)
knn_results = []


# Funções para calcular coesão e separação
def cohesion(X, labels, centroids):
    distances = np.linalg.norm(X - centroids[labels], axis=1)
    return np.sum(distances ** 2)

def separation(centroids):
    dist = pairwise_distances(centroids)
    return np.sum(dist ** 2) / 2

def cluster_entropy(labels, true_labels):
    clusters = np.unique(labels)
    total_entropy = 0.0
    for cluster in clusters:
        true_labels_in_cluster = true_labels[labels == cluster]
        class_counts = np.bincount(true_labels_in_cluster)
        probabilities = class_counts / len(true_labels_in_cluster)
        total_entropy += entropy(probabilities)
    return total_entropy / len(clusters)


## KMEANS
for n_clusters in range(2, 10):
    for max_iter in range(2, 100):

        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        kmeans.fit(X)
        center = kmeans.cluster_centers_
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Métricas Intrínsecas
        silhouette = silhouette_score(X, labels)
        total_cohesion = cohesion(X, labels, centroids)
        total_separation = separation(centroids)

        # Métricas Extrínsecas
        homogeneity = homogeneity_score(Y, labels)
        completeness = completeness_score(Y, labels)
        entropy_value = cluster_entropy(labels, Y)
        rand_index = adjusted_rand_score(Y, labels)

        # Armazenar os resultados
        knn_results.append({
            'n_clusters': n_clusters,
            'max_iter': max_iter,
            'silhouette': silhouette,
            'cohesion': total_cohesion,
            'separation': total_separation,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'entropy': entropy_value,
            'rand_index': rand_index
        })

knn_df = pd.DataFrame(knn_results)
knn_df.to_csv("knn_results.csv")



### DBSCAN

dbscan_results = []

def get_dbscan_centroids(X, labels):
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (label = -1)
    centroids = []

    for label in unique_labels:
        cluster_points = X[labels == label]  # Points in the current cluster
        centroid = np.mean(cluster_points, axis=0)  # Mean (centroid) of the cluster
        centroids.append(centroid)

    return np.array(centroids)



for samples in range(1, 5):
    for eps in range(1, 1000):
        dbscan_model = DBSCAN(min_samples=samples, eps=eps*0.00001)
        dbscan_model.fit(X)
        labels = dbscan_model.labels_
        cents = get_dbscan_centroids(X, labels)
        # Verifique o número de clusters distintos (ignora o ruído identificado como -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Continuar apenas se houver mais de um cluster
        if num_clusters > 1:
            cents = get_dbscan_centroids(X, labels)

            # Métricas Intrínsecas
            silhouette = silhouette_score(X, labels)
            total_cohesion = cohesion(X, labels, cents)
            total_separation = separation(cents)

            # Métricas Extrínsecas
            homogeneity = homogeneity_score(Y, labels)
            completeness = completeness_score(Y, labels)
            entropy_value = cluster_entropy(labels, Y)
            rand_index = adjusted_rand_score(Y, labels)

            # Armazenar os resultados
            dbscan_results.append({
                'min_samples': samples*5,
                'eps': eps*0.5,
                'silhouette': silhouette,
                'cohesion': total_cohesion,
                'separation': total_separation,
                'homogeneity': homogeneity,
                'completeness': completeness,
                'entropy': entropy_value,
                'rand_index': rand_index
            })

            # Exibir o resultado
            print(f"DBSCAN: Coesão {total_cohesion}, Separação: {total_separation}, Silhouette: {silhouette:.2f}")
        else:
            print(f"DBSCAN falhou: Apenas {num_clusters} clusters encontrados para min_samples={samples*5}, eps={eps*0.5}")

db_scan_df = pd.DataFrame(dbscan_results)
db_scan_df.to_csv("dbscan-results.csv")

### AGNES

agnes_results = []

def get_agglomerative_centroids(X, labels):
    unique_labels = np.unique(labels)  # Unique cluster labels
    centroids = []

    for label in unique_labels:
        cluster_points = X[labels == label]  # Points in the current cluster
        centroid = np.mean(cluster_points, axis=0)  # Mean (centroid) of the cluster
        centroids.append(centroid)

    return np.array(centroids)

for n_clusters in range(2, 5):
    for linkage in ['ward', 'complete', 'average', 'single']:
        agnes = AgglomerativeClustering(n_clusters, linkage=linkage)
        labels = agnes.fit_predict(X)
        cents = get_agglomerative_centroids(X, labels)

        # Métricas Intrínsecas
        silhouette = silhouette_score(X, labels)
        total_cohesion = cohesion(X, labels, cents)
        total_separation = separation(cents)

        # Métricas Extrínsecas
        homogeneity = homogeneity_score(Y, labels)
        completeness = completeness_score(Y, labels)
        entropy_value = cluster_entropy(labels, Y)
        rand_index = adjusted_rand_score(Y, labels)

        # Armazenar os resultados
        agnes_results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'cohesion': total_cohesion,
            'separation': total_separation,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'entropy': entropy_value,
            'rand_index': rand_index
        })

        print(f"AGNES: Coesão {cohesion(X, labels, cents)} Separação: {separation(cents)}")

agnes_df = pd.DataFrame(agnes_results)
agnes_df.to_csv("agnes-results.csv")
