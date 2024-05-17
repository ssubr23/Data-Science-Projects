import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load dataset
creditcard_df = pd.read_csv('./Marketing_data.csv')

# Data Overview
# CUST_ID: Identification of Credit Card holder
# BALANCE: Balance amount left in customer's account to make purchases
# BALANCE_FREQUENCY: Frequency of balance updates, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# PURCHASES: Amount of purchases made from account
# ONEOFF_PURCHASES: Maximum purchase amount done in one-go
# INSTALLMENTS_PURCHASES: Amount of purchase done in installment
# CASH_ADVANCE: Cash in advance given by the user
# PURCHASES_FREQUENCY: Frequency of purchases, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# ONEOFF_PURCHASES_FREQUENCY: Frequency of one-off purchases (1 = frequently purchased, 0 = not frequently purchased)
# PURCHASES_INSTALLMENTS_FREQUENCY: Frequency of installment purchases (1 = frequently done, 0 = not frequently done)
# CASH_ADVANCE_FREQUENCY: Frequency of cash advances
# CASH_ADVANCE_TRX: Number of cash advance transactions
# PURCHASES_TRX: Number of purchase transactions
# CREDIT_LIMIT: Credit limit of the user
# PAYMENTS: Amount of payment done by user
# MINIMUM_PAYMENTS: Minimum amount of payments made by user
# PRC_FULL_PAYMENT: Percent of full payment paid by user
# TENURE: Tenure of credit card service for the user

# Display basic information about the dataset
print(creditcard_df.info())

# Summary statistics of the dataset
print(creditcard_df.describe())

# Example queries to understand specific customer behavior
# Customer with the highest one-off purchase
print(creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == 40761.25])

# Customer with the highest cash advance
print(creditcard_df[creditcard_df['CASH_ADVANCE'] == 47137.211760000006])

# Check for missing data
sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()

# Count of missing values in each column
print(creditcard_df.isnull().sum())

# Fill missing 'MINIMUM_PAYMENTS' with mean value
creditcard_df['MINIMUM_PAYMENTS'].fillna(creditcard_df['MINIMUM_PAYMENTS'].mean(), inplace=True)

# Fill missing 'CREDIT_LIMIT' with mean value
creditcard_df['CREDIT_LIMIT'].fillna(creditcard_df['CREDIT_LIMIT'].mean(), inplace=True)

# Verify no missing values remain
sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()

# Check for duplicated entries
print(creditcard_df.duplicated().sum())

# Drop 'CUST_ID' as it is not needed for clustering
creditcard_df.drop("CUST_ID", axis=1, inplace=True)

# Visualize distributions of each feature
plt.figure(figsize=(10, 50))
for i, column in enumerate(creditcard_df.columns):
    plt.subplot(len(creditcard_df.columns), 1, i+1)
    sns.distplot(creditcard_df[column], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
    plt.title(column)
plt.tight_layout()
plt.show()

# Correlation heatmap to understand relationships between features
correlations = creditcard_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.show()

# Scaling the data
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

# Determine the optimal number of clusters using the Elbow Method
scores = []
range_values = range(1, 20)
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(creditcard_df_scaled)
    scores.append(kmeans.inertia_)

plt.plot(range_values, scores, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# Choosing 8 clusters based on the Elbow Method
kmeans = KMeans(n_clusters=8)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

# Inverse transform to understand the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=creditcard_df.columns)
print(cluster_centers_df)

# Assign cluster labels to the original dataset
creditcard_df['Cluster'] = labels

# Plot the distribution of clusters for each feature
for column in creditcard_df.columns:
    plt.figure(figsize=(20, 5))
    for cluster in range(8):
        plt.subplot(1, 8, cluster+1)
        subset = creditcard_df[creditcard_df['Cluster'] == cluster]
        subset[column].hist(bins=20)
        plt.title(f'{column} \nCluster {cluster}')
    plt.show()

# Dimensionality reduction using PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(creditcard_df_scaled)
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = labels

# Plot PCA results
plt.figure(figsize=(10, 10))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='bright')
plt.show()

# Autoencoder for dimensionality reduction
encoding_dim = 7
input_df = Input(shape=(creditcard_df_scaled.shape[1],))
x = Dense(encoding_dim, activation='relu')(input_df)
x = Dense(500, activation='relu')(x)
x = Dense(500, activation='relu')(x)
x = Dense(2000, activation='relu')(x)
encoded = Dense(10, activation='relu')(x)
x = Dense(2000, activation='relu')(encoded)
x = Dense(500, activation='relu')(x)
decoded = Dense(creditcard_df_scaled.shape[1])(x)

# Build the autoencoder model
autoencoder = Model(input_df, decoded)
encoder = Model(input_df, encoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(creditcard_df_scaled, creditcard_df_scaled, batch_size=128, epochs=25, verbose=1)

# Save the autoencoder weights
autoencoder.save_weights('autoencoder.h5')

# Encode the data using the encoder part of the autoencoder
encoded_data = encoder.predict(creditcard_df_scaled)

# Determine the optimal number of clusters for the encoded data
encoded_scores = []
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(encoded_data)
    encoded_scores.append(kmeans.inertia_)

plt.plot(range_values, encoded_scores, 'bx-')
plt.title('Finding the right number of clusters for encoded data')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# Comparing original and encoded clustering
plt.plot(range_values, scores, 'bx-', color='r', label='Original')
plt.plot(range_values, encoded_scores, 'bx-', color='g', label='Encoded')
plt.legend()
plt.show()

# Clustering the encoded data
kmeans = KMeans(n_clusters=4)
kmeans.fit(encoded_data)
labels_encoded = kmeans.labels_

# PCA for encoded data visualization
encoded_pca = PCA(n_components=2)
encoded_pca_components = encoded_pca.fit_transform(encoded_data)
encoded_pca_df = pd.DataFrame(data=encoded_pca_components, columns=['PCA1', 'PCA2'])
encoded_pca_df['Cluster'] = labels_encoded

# Plot PCA results for encoded data
plt.figure(figsize=(10, 10))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=encoded_pca_df, palette='bright')
plt.show()
