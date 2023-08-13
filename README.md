# Classification and Prediction Tasks on FIFA 18 Player Data
## I. Abstract
This project explores the uses of data mining techniques to derive useful non-trivial information from a dataset of FIFA-registered professional soccer players. Two principal tasks are explored: (1) the classification of players by their position and (2) the prediction of each player’s market value. 

For the classification tasks, four classification techniques were utilized: Decision Trees, Random Forest Classification, Naïve Bayes, and k-Nearest Neighbors. All techniques were run using Python 3.8.6. The algorithms were evaluated mainly using the accuracy metric. Clustering techniques are employed in an experiment to see if they have any meaningful impact on performance results. 

For the prediction tasks, multiple linear regression is employed and is evaluated by the Root Mean Square Error (RSME) and R2 - score metrics. The goal of this project is to be able to derive insight into whether the players are (a) played in the correct position and (b) if they are overvalued or undervalued in the transfer market.

## II. How to use the files
### Data Cleaning & Preprocessing
Provided in this repository is the raw data set, 'FIFA18playerdata.csv'. The directory called 'Data Cleaning & Preprocessing' contains the file 'cleandata.py'. This file reads in the raw data set and performs preliminary data preprocessing, creating an output CSV file called 'FIFA18playerdata_CLEANED.csv'. This is the file that is used as input for the player value prediction and player position clustering algorithms.

### Player Value Prediction
The directory 'Player Value Prediction' contains the file 'playervaluePrediction.py', The file reads in the cleaned file 'FIFA18playerdata_CLEANED.csv'.

### Position Classification
The directory 'Position Classification' contains multiple clustering algorithms, as well as one feature selection algorithm and one file to perform final classification tasks. There are four files that perform agglomerative hierarchical clustering (one for each method: average, max, min, ward). There is also one file one file for kMeans Clustering algorithm. Based on the results of the runs of agglomerative hierarchical clustering algorithms, I selected the best performing one and exported a file that included the predicted position grouping clusters. I also exported the same file from the kMeans clustering algorithm. The export files of those two algorithms are used as inputs in the classificationAlgos.py file, which compares the results of classification with and without an intermediate clustering step.   

## III. Background of the data set
The dataset used for this project was originally derived from the 2018 installment of the EA Sports FIFA video game
franchise, FIFA 19, and was sourced from Kaggle.com. The FIFA franchise is a special licensing agreement between the world governing body of soccer the Fédération Internationale de Football Association (FIFA) and game developer Electronic Arts (EA). New installments of this franchise are released every year and much of the development is built upon the collection of real-life player data by a team of 6,000 volunteer scouts.

The dataset itself contains 18,207 instances and 89 features. The features include physical attributes (height, weight, body type, preferred foot, etc.), mental attributes (work rate, composure, aggression, etc.), and skill attributes ranging from finishing to agility and dribbling. All together this data set included a comprehensive list of attributes for each player. This data set also included many attributes that were not useful for the project, such as picture hyperlinks, which were removed.

Initially the data set was split between 45 discrete and 44 numerical attributes. One of the principal challenges of this project was to reduce the dimensionality of the data, in particular the number of discrete values. Examples of preprocessing tasks were the changing of mixed string and number values such as [Acceleration : “91+2”], [Weight : “175lbs”], and [Value : “€100M”] to numeric values [Acceleration : 93], [Weight : 175], and [Value : 100000000.0].

After the removal of redundant features, the transformation of data to numeric values, and the removal of null values; the resulting data frame contained 18,147 instances and 78 features. This was only but the first stage of preprocessing and this data frame would be the basis for which all further algorithms. Depending on the algorithm, further steps of feature selection were required

## IV. Position Classification Methodology
### A. Algorithms
For positional classification, I employed four classification algorithms from Python’s scikit-learn (sklearn) library: the Decision Tree Classifier, the Random Forest Classifier, the Gaussian Naïve Bayes Classifier, and the k – Nearest Neighbor algorithm. This experiment aims to compare the classification results on the measured positions from the data set against the clustering groups determined by clustering algorithms. The clustering algorithms used are k – Means Clustering and Group Average Agglomerative Hierarchical Clustering.

### B. Preparation and Setup Methodology
As it turns out, outfield players and goalkeepers fundamentally have different assigned attributes and cause skew in the clustering algorithms as a result. Because goalkeepers are so easily identifiable and more than likely cannot play any other position on the field, I decided to drop all goalkeepers and goalkeeping attributes from the data frame – focusing my analysis solely on outfield players. In the original data set, the measured positions for outfield players were one of these 26:

LS ST RS LW LF CF 
RF RW LAM CAM RAM LM 
LCM CM RCM RM LWB LDM 
CDM RDM RWB LB LCB CB 
RCB RB

In order to make the classification and clustering process easier, I decided to transform these positions into four distinct position groupings: forwards (FWD), midfield (MID), and defense (DEF). Subsequently, I assigned a numerical value to each of the groups: DEF = 1, MID = 2, FWD = 3. The class name was “Position Grouping”.

After the removal of goalkeepers, goalkeeping attributes and non-numeric attributes, the remaining data frame included 15,871 instances and 41 numeric attributes. Because the remaining data set was still highly dimensional, a further step of feature selection was required. Feature selection was done on a correlation basis and all features that failed to meet a correlation of at least ~0.4 with regard to the Position Grouping class were dropped. The remaining 12 features were: Finishing, Volleys, Dribbling, Shot Power, Long Shots, Interceptions, Positioning, Vision, Penalties, Marking, Standing Tackle, and Sliding Tackle. These were the features on which the classifier algorithms would be trained on. Due to long run time duration, the decision was made to perform this experiment on the top 250 players.

In order to create new position groupings for the experiment, two clustering algorithms were used: k – Means and Agglomerative Hierarchical Clustering. For both clustering algorithms, it is necessary to be able to plot the instances on a Cartesian graph. Because the player data is multidimensional, it it’s difficult to plot the players on a two-dimensional plot. I felt that it was important to have as much data on each player be represented in the clustering analysis, so I utilized a Principal Component Analysis (PCA) to reduce the dimensionality to two dimensions. Due to the different scales and ranges of all of the attributes, I decided to conduct Z-normalization on the data, thereby making the PCA actually perform on the z-score values of each feature. The resulting plot is shown in Figure I below:

Figure I.
![top250players_postreduc_PCA_Scatter](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/fb3d5056-d8a3-4fea-a5a9-25aa5186747b)

PCA predicts the eigenvector which best fits the data on a variance basis – the best fit being defined as the one that maximizes the total variance between observed data points and minimizes the distance between observed points and the predicted points of the vector. As PCA computes principal components and uses them to perform a change-of-basis on the data, often some data is ignored. To check how much of the data’s variance is retained by the algorithm, I calculated the PCA explained variance ratio. For the two principal components I received the ratios of 0.747 and 0.136. This indicates that the principal components together account for 88.3% of the data, which is very good. For comparison, prior to this run of the algorithm, I tried to run the PCA without feature selection and the PCA explained variance ration was only 78.2%. Therefore, by reducing the features to only the top 12 most correlated features I was able to increase data retention by more than 10%.

Once the principal components were computed, I took the resulting plot and ran several clustering algorithms. For k-Means clustering, I selected the optimal number of clusters, k, by using the Sum of Square Errors (SSE) and the results demonstrated that three clusters were the optimal amount (see Figure II.).

Figure II.
![top250players_postreduc_kMeansClustering_SSEelbow](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/014728de-bffc-4c52-8126-09de363d8bc0)

To form the clusters, I utilized Python’s scikit-learn function, “KMeans”. The resulting cluster is demonstrated by Figure III.

Figure III.
![featreduc_top250players_kMeansClustering_k=3](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/918b7fe7-14e7-48a9-9b10-54538c416139)

Using my domain knowledge, I was able to tell that the clustering was separating players by position, with some discrepancies in the border regions.

The agglomerative hierarchical clustering approach has several methods by which to form clusters. Using sckit-learns “AgglomerativeClustering” function, I ran four hierarchical clustering’s with the following linkage methods: single (MIN), complete (MAX), average (Group Average), and the Ward method. After generating dendrograms for each method, the best results were from the Group Average method. The resulting dendrograms and clustering results are shown in Figure IV. and Figure V. respectfully.

Figure IV.
![featreduced_top250_Dendogram_AVGmethod](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/aca4aa3f-d2a9-4380-9738-fda615e74648)

Figure V.
![featreduc_top250_hierarchical_AVGmethod_n=3](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/6e276c34-475b-4fda-b241-989228ce7eaa)

Both algorithms produced interesting clusters which were not well separated. Just by looking at the plots of the two clustering algorithms, I noticed interesting variations in the border regions of each cluster, which made me curious as to what were the defining features of each cluster. I generated parallel coordinate visualizations, which can be seen in Figure I and Figure II in the Appendix. The results from the k-Means and agglomerative clustering were appended to the data frame which would be used for the classification algorithms. The resulting data used to train the algorithms was split on an 80/20 basis – 80% of the data was used to train the model and 20% was used to test it. The main purpose of this experiment is to test how clustering can improve player classification; therefore, the
train/test split is one of the controlled variables. Altogether, there were four classification models run on three different sets of classes, totaling 12 total classification algorithms being run.

### Evaluation Metrics
After running each of the classification algorithms on both the measured position groupings and those generated by the clustering algorithms, I generated confusion matrices and score reports for each model. The confusion matrices helped me identify where the misclassification errors occurred. The score reports that were generated included the accuracy, precision, and f1-score. Although the precision and f1-scores were generated, I was mostly concerned with the accuracy. There was little class imbalance for the classifiers, therefore I felt that accuracy would be a suitable metric. Additionally, the nature of this experiment was to determine how, ceteris paribus, clustering the data beforehand could impact classification accuracy results. Therefore, once all the algorithms were tested, I compared the accuracy results side by side.

## V. Player Value Prediction Methodology
### A. Algorithms
In order to predict player market value, I employed three different models: multiple linear regression algorithm, Random Forest Regresion, and Gradient Boostong Regression. The packages utilized were from the Python scikit-learn library ensemble.

### B. Preparation and Setup Methodology
To perform this task, I utilized a few different methods of preparing the data set to be run by the linear regression algorithm. The data frame used was the one generated after the initial preprocessing and cleanup stage discussed in part II of this paper.

For training the models, I decided to train and test them using 80/20, 66/33, and 50/50 splits, ultimately comparing the results of each. The first model run is on the full dataset with no feature selection. As mentioned previously, goalkeepers fundamentally have different attributes in this data set. As such, under the 26 numeric position attributes which are recorded for each outfield player, goalkeepers have zero values. The same holds true for outfield players and their goalkeeping attributes. With this is mind, as part of my experiment I separate the data set into outfield player and goalkeeping player data frames and run linear regression on them separately, aggregating the results after the fact. Across three different training/testing splits and three linear regression models per split, I ran nine different regression models.

### C. Evaluation Metrics
To evaluate the results of each regression model, I utilized two metrics: Root Mean Square Error (RMSE) and the R-squared score. The RMSE metric served as insight into how far apart the predicted valuations and measured valuations were on average. The R-squared metric helped me understand how well the models fit the data sets. Additionally, after running each model and producing the metric scores mentioned above, I also printed out a small sample of players and compared the predicted values to the measured values; using my domain knowledge to determine if the predicted values were reasonable.

## VI. Results
### A. Position Classification Task
Overall, the classification algorithms performed significantly better when trained on the classes defined by the clustering algorithms than those measured in the data set. The performance reports of the classification algorithms on the measured and clustered position classes can be seen in Table I. and Table II., respectfully.

![image](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/89aed3e5-ef4b-4a6a-9680-ae2d27f175e0)

![image](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/e99b400f-9e93-4105-8322-40dca990788c)

As is evident, the performances of the classification algorithms are significantly better when trained on clustered position classes than when trained on the measured data set classes. The likely reason for this is that the clustering algorithms provide more structure and better explain the data at hand. The clustered classes are more correlated to the measured attributes than the classes given. The results demonstrate that applying clustering as part of a preprocessing step is beneficial, especially when you have a such a highly dimensional data set as the one used in this paper.

As far as the real-world implications of these results, by reconfiguring the classes using clustering, I was able to improve the veracity of the training data. If I am a player scout or training staff on a professional team, for instance, I can use any of the classification algorithms trained on this improved data to classify with a high degree of accuracy the most natural position for any player that comes into the team to play in. This is very useful when dealing with players that have played in different positions throughout their career with varying levels of success.

### B. Player Value Prediction Task
The results of the nine linear regression runs are demonstrated in Table III.

![image](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/cf3284ea-0cb3-4b0d-9623-73c8c87227ce)

As is seen by the results, the best model to run the algorithm on the full data set is the one using the 80/20 split train/test split. Using this split of data is the RMSE is at a minimum and the R – squared score is closest to one, thus implying that it is the model which best fits the data. However, when I looked at the predicted values for the players, I saw that the goalkeepers’ values were predicted to be significantly lower than measured. I expected this to be the case and it was the original motivation for running these algorithms on outfield players and goalkeepers separately. As goalkeepers only make up 10% of the entire player data set, all models performing on solely the goalkeepers run the risk of severely underfitting the data. This was demonstrated in the R-squared results for the 80/20 and 66/33 splits – both 0.44 and 0.49 respectively. However, when splitting the data on a 50/50 basis, I was able to significantly increase the R-squared score to 0.70. Comparing the weighted average RSME scores for outfield and goalkeeper data sets, as well as the results from the full player data sets, I heuristically came to the conclusion that utilizing the results from the 50/50 split “Outfield” and “Goalkeeper” models would achieve the best insight gain.

Upon making this decision, I compared the predicted values with the measured values of the top 10 players in order to determine if players were overvalued or undervalued. The results can be seen in Table IV and Table V.

![image](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/933d24e9-b6d9-4778-935e-54de940b646e)

![image](https://github.com/wgemba/fifa-player-data-mining/assets/134420287/f32b8918-98dc-447f-953d-ad7431826914)

## VII. Conclusion
In this project I explored a few applications of data mining techniques for the purposes of accurately classifying soccer players based on their positions, as well as accurately
predicting their true value in the transfer market. The classification experiment focused on the application of clustering as an additional preprocessing step in order to boost the performance of classification algorithms. The results clearly demonstrated that this clustering is beneficial for classification and the experiment achieved what it set out to accomplish. However, further avenues for work could be the analysis of how one might be able to use pruning methods to avoid over-fitting, as well as cross validation for hyper-parameter selection. These methods could be applied to each individual classifier to boost their individual performances. Due to the nature of this experiment, as well as the high computational cost of running so many different algorithms, it was not applicable to examine these effects.

From the results of the liner regression models we can see a lot of very interesting results for player value. The results show a good mixture of players that are overvalued and undervalued. For further research it would be interesting to see how attributes beyond the realm of this data set influenced the measured values. For one, it is important to keep in mind that professional sports teams are also businesses and generate a lot of revenue from player shirt and memorabilia sales. For instance, Lionel Messi, arguably the most famous player in the world, generates hundreds of millions of euros in yearly sales revenue for his club, FC Barcelona. Players with this kind of brand positioning are inevitably valued higher in the transfer market. Therefore, I would be interested in seeing how an additional relevant data would influence the prediction results.
