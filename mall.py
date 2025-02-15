import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Mall_Customers.csv"  # Update with the correct file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values and handle them (only for numeric columns)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Select relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Apply K-means clustering with the optimal k (replace with your chosen k)
optimal_k = 5  # Determine from the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

data['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Customer Segmentation using K-means Clustering')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()

# Analyze the clusters (only numeric columns to avoid errors)
numeric_data = data.select_dtypes(include=[np.number])
cluster_analysis = numeric_data.groupby('Cluster').mean()
print("Cluster Analysis:")
print(cluster_analysis)

# Save results excluding the specified columns
data_without_columns = data.drop(columns=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
data_without_columns.to_csv('customer_segmentation_results.csv', index=False)
print("Clustering results without specified columns saved to customer_segmentation_results_without_columns.csv")
