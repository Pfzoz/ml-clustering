import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data.csv")

# Display the first 5 rows of the data
print(data.head())

# Assuming that the first two columns of the dataset are the features for clustering
X = data.iloc[:, :2].values  # Selecting only the first two columns for clustering

# Visualize the data points before clustering
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], cmap="rainbow")
plt.title("Data points")
plt.show()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, max_iter=300, random_state=0)
kmeans.fit(X)

# Get the cluster centers and labels
centro = kmeans.cluster_centers_
labels = kmeans.labels_

# Print cluster centers
print("Cluster Centers:\n", centro)

# Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.scatter(centro[:, 0], centro[:, 1], s=300, c='black', marker='x')  # Mark cluster centers
plt.title("Clusters with KMeans")
plt.show()
