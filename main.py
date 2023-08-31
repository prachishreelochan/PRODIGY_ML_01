import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Example data (Replace this with your actual customer data)
data = np.array([
    [100, 5],   # Customer 1: Total spending, Number of purchases
    [50, 3],    # Customer 2: Total spending, Number of purchases
    [300, 10],  # Customer 3: Total spending, Number of purchases
    # Add more customer data here
])
# Set the number of clusters (K)
num_clusters = 3
# Initialize the KMeans object
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster assignments for each customer
cluster_assignments = kmeans.predict(data)

# Get the cluster centers (representative points of each cluster)
cluster_centers = kmeans.cluster_centers_
# Separate the clustered data based on the assigned cluster
clustered_data = [data[cluster_assignments == i] for i in range(num_clusters)]

# Plot the clustered data
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow']  # Add more colors if needed
for i, cluster_data in enumerate(clustered_data):
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i+1}')

# Plot the cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=200, label='Cluster Centers')

# Add labels and legend
plt.xlabel('Total Spending')
plt.ylabel('Number of Purchases')
plt.legend()
plt.title('Customer Segmentation with K-means Clustering')
plt.grid(True)
plt.show()

