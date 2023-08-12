"""
Author: William Gemba

This is a file to perform Data Mining Tasks of the cleaned FIFA 18 Player Data Set.

"""
import os
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode
import plotly.io as pio
import cufflinks as cf
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from adjustText import adjust_text

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)
cf.go_offline()

### Set Plotly Renderer to a Default Value ###
pio.renderers.default = "browser"

### Set Display Options for Panda ###
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

os.chdir('C:/Users/willi/Documents/Data Science Practice Projects/Fordham Projects/Data_Mining_Final_Project_Clustering_Classification/')
df = pd.read_csv('Project_v2/FIFA18playerdata_CLEANED_featurereduc.csv', index_col='Name')

df.head()
df.info()

dfcopy = df.copy()
dfcopy = dfcopy.head(250)
dfcopy = dfcopy.drop(columns=['Position Grouping'], inplace=False)

### To be used as labels for points later ###
label_names = dfcopy.index.tolist()

df_na_zscores = dfcopy.transform(lambda x: (x - x.mean()) / x.std())

print(dfcopy[:3])

### Make the Clustering Model Based Upon Two Dimensions Using PCA ###

x = df_na_zscores.values # returns a numpy array
minmax_scaler = preprocessing.MinMaxScaler()  # sklearn scaler
x_scaled = minmax_scaler.fit_transform(x)  # fit scaler
X_norm = pd.DataFrame(x_scaled)

# turns a set of correlated features into a set of linearly uncorrelated ones, capturing the greatest variablity between features
pca = PCA(n_components=2) # 2-dimensional PCA
reduced = pd.DataFrame(pca.fit_transform(X_norm)) # new dataframe
reduced.columns = ['Principal Component 1', 'Principal Component 2']

print('\n')
print('Data Retention is : ')
print(pca.explained_variance_ratio_)
print('\n')

# Visualize un-Clustered Scatter

plt.figure(figsize=(10,7))
plt.scatter(reduced['Principal Component 1'], reduced['Principal Component 2'], cmap='rainbow')
plt.title('Cartesian Scatter Plot of Players', fontsize=25)
plt.xlabel("Principal Component 1", fontsize=20)
plt.ylabel("Principal Component 2", fontsize=20)
plt.show()


### kMeans Clustering ###
## Step 1: Determine Number(k) of Iterations using SSE v Clusters plotting

clustrange = range(1,11)
clusterrors = []

for numclust in clustrange:
    clusters = KMeans(numclust)
    clusters.fit(reduced)
    clusterrors.append(clusters.inertia_)

cluster_dframe = pd.DataFrame({"numbclust":clustrange, "clusterrors": clusterrors})

print (cluster_dframe)

# Visualize SSE vs. Number of Clusters

sns.set(style="white")
plt.figure(figsize=(12,6))
plt.plot(cluster_dframe.numbclust, cluster_dframe.clusterrors, marker = "o")
plt.tick_params(labelsize = 15)

plt.title('Elbow Method for Optimal k', fontsize=25)
plt.xlabel("Number of Clusters", fontsize=20)
plt.ylabel("SSE", fontsize=20)
plt.show()


reducedcopy1 = reduced.copy()
reducedcopy2 = reduced.copy()
## Step 2- After Step 1 demonstrates that 3,4,5 iterations could be the "elbow", run iterations.
kmeans = KMeans(n_clusters=3)  # Number of clusters
kmeans = kmeans.fit(reduced)  # Fitting the input data
labels = kmeans.predict(reduced)  # Getting the cluster labels
centroid = kmeans.cluster_centers_  # Centroid values
clusters = kmeans.labels_.tolist()  # create a list to add column to original df

reduced['cluster'] = clusters
reduced['name'] = label_names
reduced.columns = ['Principal Component 1', 'Principal Component 2', 'cluster', 'name']
print(reduced[:3])

samplesCentroids = centroid[labels]
print(samplesCentroids)

## Step 3 - Plot Results

plt.figure(figsize=(10,7))
plt.scatter(reduced['Principal Component 1'], reduced['Principal Component 2'], c=labels, cmap='rainbow')
plt.scatter(centroid[:,0], centroid[:,1], marker= '+', s =150, linewidths=2, zorder=10, c = 'black')
reduced['name'] = label_names

plottexts = []
for x,y,s in zip(reduced['Principal Component 1'], reduced['Principal Component 2'], reduced['name']):
    plottexts.append(plt.text(x,y,s))
adjust_text(plottexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.title('kMeans Clustering (k = 3)', fontsize=25)
plt.xlabel("Principal Component 1", fontsize=20)
plt.ylabel("Principal Component 2", fontsize=20)
plt.show()

df_withClusters = dfcopy.copy()

df_withClusters['kMeansClusters(k=3)']= pd.Series(labels, index=df_withClusters.index)
print(df_withClusters[:10])

## Must add each clustering results for each k
# k = 4
kmeans = KMeans(n_clusters=4)  # Number of clusters
kmeans = kmeans.fit(reducedcopy1)  # Fitting the input data
labels = kmeans.predict(reducedcopy1)  # Getting the cluster labels

df_withClusters['kMeansClusters(k=4)']= pd.Series(labels, index=df_withClusters.index)
print(df_withClusters[:10])

# k = 5
kmeans = KMeans(n_clusters=5)  # Number of clusters
kmeans = kmeans.fit(reducedcopy1)  # Fitting the input data
labels = kmeans.predict(reducedcopy2)  # Getting the cluster labels

df_withClusters['kMeansClusters(k=5)']= pd.Series(labels, index=df_withClusters.index)
print(df_withClusters[:10])

df_withClusters.head()
df_withClusters.info()

#df_withClusters.to_csv(r'C:/Users/willi/Documents/1 - FORDHAM/GRADUATE/Academic/CISC 5790 - Data Mining/Final Project/Project_v2/FIFA18playerdata_CLEANED_top250Players_kMeaned_postreduc.csv', index = True)

def mapTarget(val):
    if val == 0:
        return "C0 (MID)"
    elif val == 2:
        return "C1 (DEF)"
    elif val == 1:
        return "C2 (FWD)"


df_withClusters['target_name'] = df_withClusters['kMeansClusters(k=3)'].apply(mapTarget)

del df_withClusters['kMeansClusters(k=3)']
del df_withClusters['kMeansClusters(k=4)']
del df_withClusters['kMeansClusters(k=5)']

print(df_withClusters)

parallel1 = parallel_coordinates(df_withClusters, 'target_name', color=['r', 'g', 'b'])
plt.title('Feature Variance by Clusters (kMeans Clustering, k=3)', fontsize=25)
plt.show()