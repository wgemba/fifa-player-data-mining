"""
Author: William Gemba

This is a file to perform Data Mining Tasks of the cleaned FIFA 18 Player Data Set.

Python 3.8.6
"""
import os
import scipy.cluster.hierarchy as shc
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
import plotly.io as pio
import cufflinks as cf
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
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

## MAX Method ###
plt.figure(figsize=(10,7))
plt.title("Player PCA Dendogram (MAX method)")
dend = shc.dendrogram(shc.linkage(reduced, method = 'complete'),labels= label_names)
plt.show()

agglo_cluster_max = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'complete')

plt.figure(figsize=(10,7))
plt.scatter(reduced['Principal Component 1'], reduced['Principal Component 2'], c=agglo_cluster_max.fit_predict(reduced), cmap='rainbow')

reduced['name'] = label_names

plottexts = []
for x,y,s in zip(reduced['Principal Component 1'], reduced['Principal Component 2'], reduced['name']):
    plottexts.append(plt.text(x,y,s))
adjust_text(plottexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.title('Agglomerative Hierarchical Clustering (MAX Method)', fontsize=25)
plt.xlabel('Principal Component 1', fontsize =20)
plt.ylabel('Principal Component 2', fontsize =20)
plt.show()