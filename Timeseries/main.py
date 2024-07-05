import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data from a CSV file
data = pd.read_csv('data.csv')

# Display the first few rows of the dataframe
print("First few rows of the data:")
print(data.head())

# Check for missing values
print("\nMissing values in the data:")
print(data.isnull().sum())

# Fill or drop missing values
data = data.dropna()  # or use data.fillna(method='ffill') for forward fill

# Convert timestamp to datetime if it's not already
data['_time'] = pd.to_datetime(data['_time'])

# Set timestamp as the index
data.set_index('_time', inplace=True)

# Plot the data to visualize trends
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['_value'])
plt.title('Data over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Describe the data
print("\nData description:")
print(data.describe())

# Check for correlations
print("\nData correlation:")
print(data.corr())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['_value']])

# Determine the number of clusters using the elbow method
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means clustering
optimal_clusters = 3  # This should be determined from the elbow plot
kmeans = KMeans(n_clusters=optimal_clusters)
data['cluster'] = kmeans.fit_predict(scaled_data)

# Plot the clustered data
plt.figure(figsize=(10, 5))
plt.scatter(data.index, data['value'], c=data['cluster'], cmap='viridis')
plt.title('Clustered Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Save the clustered data to a new CSV file
data.to_csv('clustered_data.csv')

print("Clustering complete. The clustered data has been saved to 'clustered_data.csv'.")
