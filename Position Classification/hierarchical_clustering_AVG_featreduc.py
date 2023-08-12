"""
Author: William Gemba

This is a file to perform Data Mining Tasks of the cleaned FIFA 18 Player Data Set.

Python 3.8.6
"""
import os
import scipy.cluster.hierarchy as shc
import pandas as pd
from pandas.plotting import parallel_coordinates
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

print(df.head())
print(df.info())

dfcopy = df.copy()
dfcopy = dfcopy.head(250)
dfcopy = dfcopy.drop(columns=['Position Grouping'], inplace=False)

### To be used as labels for points later ###
label_names = dfcopy.index.tolist()

# Normalize numeric data
df_na_zscores = dfcopy.transform(lambda x: (x - x.mean()) / x.std())

print(df_na_zscores[:3])

### Make the Clustering Model Based Upon Two Dimensions Using PCA ###

x = df_na_zscores.values # returns a numpy array
minmax_scaler = preprocessing.MinMaxScaler()  # sklearn scaler
x_scaled = minmax_scaler.fit_transform(x)  # fit scaler
X_norm = pd.DataFrame(x_scaled)

# Turns a set of correlated features into a set of linearly uncorrelated ones, capturing the greatest variablity between features
pca = PCA(n_components=2) # 2-dimensional PCA
reduced = pd.DataFrame(pca.fit_transform(X_norm)) # new dataframe
reduced.columns = ['Principal Component 1', 'Principal Component 2']

print('\n')
print('PCA Variance Retention is : ')
print(pca.explained_variance_ratio_)
print('\n')

# Visualize un-Clustered Scatter

plt.figure(figsize=(10,7))
plt.scatter(reduced['Principal Component 1'], reduced['Principal Component 2'], cmap='rainbow')
plt.title('Cartesian Scatter Plot of Players', fontsize=25)
plt.xlabel("Principal Component 1", fontsize=20)
plt.ylabel("Principal Component 2", fontsize=20)
plt.show()

## AVG Method ###
plt.figure(figsize=(10,7))
plt.title("Player PCA Dendogram (AVG method)")
dend = shc.dendrogram(shc.linkage(reduced, method = 'average'),labels= label_names)
plt.show()

agglo_cluster_avg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'average')

plt.figure(figsize=(10,7))
plt.scatter(reduced['Principal Component 1'], reduced['Principal Component 2'], c=agglo_cluster_avg.fit_predict(reduced), cmap='rainbow')

reduced['name'] = label_names

plottexts = []
for x,y,s in zip(reduced['Principal Component 1'], reduced['Principal Component 2'], reduced['name']):
    plottexts.append(plt.text(x,y,s))
adjust_text(plottexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.title('Agglomerative Hierarchical Clustering (AVG Method)', fontsize=25)
plt.xlabel('Principal Component 1', fontsize =20)
plt.ylabel('Principal Component 2', fontsize =20)
plt.show()

# Add clusterings as new attribute
df_withClusters = dfcopy.copy()

df_withClusters['Hierachical_Clustering_AVG']= pd.Series(agglo_cluster_avg.labels_, index=df_withClusters.index)
print(df_withClusters[:10])

"""['Finishing','Volleys','Dribbling','ShotPower','LongShots','Interceptions',
                                     'Positioning','Vision','Penalties','Marking','StandingTackle','SlidingTackle']"""

def mapTarget(val):
        if val == 0:
            return "C0 (MID)"
        elif val == 1:
            return "C1 (DEF)"
        elif val == 2:
            return "C2 (FWD)"

df_withClusters['target_name'] = df_withClusters['Hierachical_Clustering_AVG'].apply(mapTarget)

del df_withClusters['Hierachical_Clustering_AVG']

print(df_withClusters)

#df_withClusters.to_csv(r'C:/Users/willi/Documents/1 - FORDHAM/GRADUATE/Academic/CISC 5790 - Data Mining/Final Project/Project_v2/FIFA18playerdata_CLEANED_top250Players_HierarchicalAVG_postreduc.csv', index = True)

# Visualize feature variance by cluster

parallel1 = parallel_coordinates(df_withClusters, 'target_name', color=['r', 'g', 'b'])
plt.title('Feature Variance by Clusters (Hierarchical Clustering)', fontsize= 25)
plt.show()

