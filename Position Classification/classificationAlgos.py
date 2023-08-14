"""
Author: William Gemba

This is a file to perform Data Mining Tasks of the cleaned FIFA 18 Player Data Set.

Python 3.8.6
"""
import os
import numpy as np
import scipy.cluster.hierarchy as shc
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
import plotly.io as pio
import cufflinks as cf
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from  sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree, naive_bayes

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

os.chdir('<Insert PATH Here>')
df = pd.read_csv('FIFA18playerdata_CLEANED_featurereduc.csv', index_col='Name')

df.head()
df.info()

dfcopy = df.copy()
dfcopy = dfcopy.head(250)

X = dfcopy[['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision',
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle']]

y = dfcopy['Position Grouping']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

classifiers = []

model1 = tree.DecisionTreeClassifier()
classifiers.append(model1)

model2 = RandomForestClassifier()
classifiers.append(model2)

model3= naive_bayes.GaussianNB()
classifiers.append(model3)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model4 = KNeighborsClassifier(n_neighbors=8) #Based on Error Rate K value
model4.fit(X_train, y_train)
classifiers.append(model4)

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print('The Accuracy of %s is %s' %(clf,accu))
    conmat = confusion_matrix(y_test, y_pred)
    print('The Confusion Matrix of %s is %s\n' %(clf,conmat))
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_test, y_pred, target_names=target_names))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='blue', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value (Using Class from Data Set)')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


""" Classify on Clusters"""
### KMeans ###

df_kmeans = pd.read_csv('Project_v2/PostFeatureReducion/FIFA18playerdata_CLEANED_top250Players_kMeaned_postreduc.csv', index_col='Name')

df_kmeans.head()
df_kmeans.info()

dfkmeans3 = df_kmeans.filter(['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision',
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle', 'kMeansClusters(k=3)'], axis=1)
dfkmeans4 = df_kmeans.filter(['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision', # Never Used
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle', 'kMeansClusters(k=4)'], axis=1)
dfkmeans5 = df_kmeans.filter(['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision', # Never Used
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle', 'kMeansClusters(k=5)'], axis=1)

Xk3 = dfkmeans3[['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision',
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle']]

yk3 = dfkmeans3['kMeansClusters(k=3)']

Xk3_train, Xk3_test, yk3_train, yk3_test = train_test_split(Xk3,yk3, test_size=0.2, random_state=42)

classifiers_kM = []

modelkM1 = tree.DecisionTreeClassifier()
classifiers_kM.append(modelkM1)

modelkM2 = RandomForestClassifier()
classifiers_kM.append(modelkM2)

modelkM3= naive_bayes.GaussianNB()
classifiers_kM.append(modelkM3)

scaler.fit(Xk3_train)
Xk3_train = scaler.transform(Xk3_train)
Xk3_test = scaler.transform(Xk3_test)
modelkM4 = KNeighborsClassifier(n_neighbors=6) #Based on Error Rate K value
modelkM4.fit(Xk3_train, yk3_train)
classifiers_kM.append(modelkM4)

for clf in classifiers_kM:
    clf.fit(Xk3_train, yk3_train)
    yk3_pred = clf.predict(Xk3_test)
    accukM = accuracy_score(yk3_test, yk3_pred)
    print('The Accuracy of %s is %s' %(clf,accukM))
    conmatkM = confusion_matrix(yk3_test, yk3_pred)
    print('The Confusion Matrix of %s is %s\n' %(clf,conmatkM))
    target_names_kM = ['class 0', 'class 1', 'class 2']
    print(classification_report(yk3_test, yk3_pred, target_names=target_names_kM))

    errorkM = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(Xk3_train, yk3_train)
        pred_i = knn.predict(Xk3_test)
        errorkM.append(np.mean(pred_i != yk3_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), errorkM, color='blue', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value ((Using Class from kMeans Clustering))')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

### Agglomerative Hierarchical MAX ###

df_agglo = pd.read_csv('Project_v2/Hierarchical Clustering/FIFA18playerdata_CLEANED_top250Players_HierarchicalAVG_postreduc.csv', index_col='Name')

df_agglo.head()
df_agglo.info()

df_agglo = df_agglo.filter(['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision',
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle', 'Hierarchical_Clustering_AVG'], axis=1)

Xa = df_agglo[['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision',
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle']]

ya = df_agglo['Hierarchical_Clustering_AVG']

Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa,ya, test_size=0.2, random_state=42)

classifiers_agglo = []

model1a = tree.DecisionTreeClassifier()
classifiers_agglo.append(model1a)

model2a = RandomForestClassifier()
classifiers_agglo.append(model2a)

model3a= naive_bayes.GaussianNB()
classifiers_agglo.append(model3a)

scaler.fit(Xa_train)
Xa_train = scaler.transform(Xa_train)
Xa_test = scaler.transform(Xa_test)
model4a = KNeighborsClassifier(n_neighbors=6) #Based on Error Rate K value
model4a.fit(Xa_train, ya_train)
classifiers_agglo.append(model4a)

for clf in classifiers_agglo:
    clf.fit(Xa_train, ya_train)
    ya_pred = clf.predict(Xa_test)
    accu_a = accuracy_score(ya_test, ya_pred)
    print('The Accuracy of %s is %s' %(clf,accu_a))
    conmat_a = confusion_matrix(ya_test, ya_pred)
    print('The Confusion Matrix of %s is %s\n' %(clf,conmat_a))
    target_names_a = ['class 0', 'class 1', 'class 2']
    print(classification_report(ya_test, ya_pred, target_names=target_names_a))

    error_a = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(Xa_train, ya_train)
        pred_i = knn.predict(Xa_test)
        error_a.append(np.mean(pred_i != ya_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error_a, color='blue', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value (Using Class from Hierarchical Clustering (AVG))')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
