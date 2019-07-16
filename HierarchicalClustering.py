# Kirill's Notes on Hierarchical Clustering from a custom data set
# Here, clusting is used to find most distinctive cluster of vehicles

import numpy as np
import pandas as pd
import scipy
import pylab
import scipy.cluster.hierarchy
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler # This is for normalizing the feature set, it translates each one of the features one by one so that it is between 0 and 1.

# Load the data from csv file
pdf = pd.read_csv('cars.csv')
# Inspect the dataset
print(pdf.head(10))
print ("The number of rows and colums respectively is: ", pdf.shape)
# Remove rows with null values
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
# Inspect the new altered data dataframe
print(pdf.head(10))
print ("The new number of rows is: ", pdf.shape)
# Create a feature set
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
# Normalize the feature set using MinMaxScalar which will scale each feature in a given range (default range is (0,1))
x = featureset.values # This returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx [0:5])
# Agglomeratie clustering requires that at each iteration, the algorithm must update the distance matrix to reflect the
# the distance of new clusters with the remaining clusters in the "forest"
# Use Scipy to cluster the dataset by first calculating the distance matrix
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
Z = hierarchy.linkage(D, 'complete')
# Creeate a cutting line
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)
# Determine the number of clusters directly
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)
# Plot the Dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.show()
