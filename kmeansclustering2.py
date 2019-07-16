# Kirill's Notes on K-Means Clustering from a custom data set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# laod data from csv
cust_df = pd.read_csv("cs.csv")
print(cust_df.head())

#                        Pre-processing
# Address in this dataset is a categorical variable. k-means algorithm
# isn't directly applicable to categorical variables because Euclidean
# distance function isn't really meaningful for discrete variables. So,
# lets drop this feature and run clustering.
df = cust_df.drop('Address', axis=1)
print(df.head())

#               Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the
# first place? Normalization is a statistical method that helps mathematical-based
# algorithms to interpret features with different magnitudes and distributions equally.
# We use StandardScaler() to normalize our dataset.
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

# Apply k-means on our dataset, and take look at cluster labels.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# Assign labels to each row in the dataframe.
df["Clus_km"] = labels
print(df.head(5))

# Check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

# look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
