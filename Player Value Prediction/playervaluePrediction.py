"""
Author: William Gemba

Code for Player Market Value Predictions

Python 3.8.6
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns

### Set Display Options for Pandas ###
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def formatInt(val):
    val = int(val)
    return val

os.chdir('C:/Users/willi/Documents/Data Science Practice Projects/Fordham Projects/Data_Mining_Final_Project_Clustering_Classification/')
df = pd.read_csv('FIFA18playerdata_CLEANED.csv', index_col='Name')

df.head()
df.info()

df.drop(columns=['Position'])


### Create separate data frames for Goalkeepers as they will be evaluated separatly
df_GK = df.drop(columns=['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB', 'LDM', 'CAM', 'CDM',
                     'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM', 'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB', 'Position Grouping'], inplace=False)

df_GK_attributes_only = df_GK.filter(['Age','Height (cm)','Weight (lbs)', 'Overall','Potential','International Reputation','GKDiving',
                                     'GKHandling','GKKicking', 'GKPositioning', 'GKReflexes', 'Value (Euros)', 'Weekly Wage (Euros)'], axis=1)
df_GK_attributes_only.head()
df_GK_attributes_only.info()

## Drop GK for test
index_names_GK = df[df['Position Grouping'] == 'GK'].index
df_field_players = df.drop(index_names_GK, inplace = False)

df_field_players = df_field_players.drop(columns=['GKDiving', 'GKHandling','GKKicking', 'GKPositioning', 'GKReflexes'])

df_field_attributes = df_field_players.filter(['Age','Height (cm)','Weight (lbs)', 'Overall','Potential','International Reputation', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                                               'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
                                               'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                                               'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle','Value (Euros)', 'Weekly Wage (Euros)'], axis=1)
df_field_attributes.head()
df_field_attributes.info()

print(df_field_attributes[:3])

##################################### Visualizations

# Show Value Distribution
plt.figure(figsize= (10,7))
plt.hist(df_field_attributes['Value (Euros)'])
plt.title('Distribution of Market Valuations', fontsize = 25)
plt.show()


# Overall vs Value
plt.figure(figsize= (10,7))
plt.scatter(df_field_attributes['Overall'],df_field_attributes['Value (Euros)'])
plt.title('Overall vs Value', fontsize = 25)
plt.xlabel('Overall', fontsize = 20)
plt.ylabel('Value (Euros)', fontsize = 20)
plt.show()

### Feature Selection Correlation Matrix

corr_matrix = df_field_attributes.corr().abs()
print(corr_matrix)
print('\n')
print(corr_matrix['Value (Euros)'], )

plt.figure(figsize=(10,7))
ax = sns.heatmap(corr_matrix)
plt.title('Correlation Matrix', fontsize = 25)
plt.show()

### Modeling
# Pre-feature reduction
selected_numeric_training_vals = ['Age','Height (cm)','Weight (lbs)', 'Overall','Potential','International Reputation', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                                               'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
                                               'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
                                               'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']

Xfield1 = df_field_attributes[selected_numeric_training_vals]
yfield1 = df_field_attributes['Value (Euros)']

print(Xfield1.shape, yfield1.shape, '\n')

Xfield_train, Xfield_test, yfield_train, yfield_test = skl.model_selection.train_test_split(Xfield1, yfield1, test_size=0.33, random_state=42)

print(Xfield_train.shape, yfield_train.shape ,'\n')
print(Xfield_test.shape, yfield_test.shape ,'\n')

# Feature reduction based on low correlation to player value
selected_numeric_training_vals2 = ['Overall','Potential','International Reputation', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                                               'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
                                               'Reactions', 'ShotPower', 'Stamina', 'LongShots','Positioning', 'Vision', 'Penalties', 'Composure']

Xfield2 = df_field_attributes[selected_numeric_training_vals2]
yfield2 = df_field_attributes['Value (Euros)']

print(Xfield2.shape, yfield2.shape, '\n')

Xfield_train2, Xfield_test2, yfield_train2, yfield_test2 = skl.model_selection.train_test_split(Xfield2, yfield2, test_size=0.33, random_state=42)

print(Xfield_train2.shape, yfield_train2.shape ,'\n')
print(Xfield_test2.shape, yfield_test2.shape ,'\n')

# Linear Regression prediction without feature redcution
model1 = LinearRegression()
model1.fit(Xfield_train, yfield_train)

print('Intercept: \n', model1.intercept_)
print('Coefficients: \n', model1.coef_)

Y_pred = model1.predict(Xfield_test)

print('\n')
print('========== Results Evaluation (Linear Regression without Feature Reduction) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield_test, Y_pred)))
r2 = r2_score(yfield_test, Y_pred)
print('R-Squared Score: ', r2)
print('======================================================================================\n')

#Linear Regression prediction with feature reduction
model2 = LinearRegression()
model2.fit(Xfield_train2, yfield_train2)

print('Intercept: \n', model2.intercept_)
print('Coefficients: \n', model2.coef_)

Y_pred = model2.predict(Xfield_test2)

print('\n')
print('========== Results Evaluation (Linear Regression with Feature Reduction) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield_test2, Y_pred)))
#print('RMSLE:', np.sqrt(mean_squared_log_error(yfield_test, pred)))
r2 = r2_score(yfield_test2, Y_pred)
print('R-Squared Score: ', r2)
print('===================================================================================\n')

yfield_pred_all = model1.predict(Xfield1)

print('Total preditions: ' ,len(yfield_pred_all))

## Run Random Forest Regressor without feature reduction

model3 = RandomForestRegressor()
model3.fit(Xfield_train, yfield_train)
yfield_pred_rf = model3.predict(Xfield_test)

print('\n')
print('========== Results Evaluation (Random Forest without feature reduction) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield_test, yfield_pred_rf)))
r2 = r2_score(yfield_test, yfield_pred_rf)
print('R-Squared Score: ', r2)
print('==================================================================================\n')

## Run Random Forest Regressor with feature reduction

model4 = RandomForestRegressor()
model4.fit(Xfield_train2, yfield_train2)
yfield_pred_rf = model4.predict(Xfield_test2)

print('\n')
print('========== Results Evaluation (Random Forest without feature reduction) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield_test2, yfield_pred_rf)))
r2 = r2_score(yfield_test2, yfield_pred_rf)
print('R-Squared Score: ', r2)
print('==================================================================================\n')

## Run Gradient Boosting Regressor without feature reduction

model5 = GradientBoostingRegressor()
model5.fit(Xfield_train, yfield_train)
yfield_pred_gbr = model5.predict(Xfield_test)

print('\n')
print('========== Results Evaluation (Gradient Boosting Regressor without feature reduction) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield_test, yfield_pred_gbr)))
r2 = r2_score(yfield_test, yfield_pred_gbr)
print('R-Squared Score: ', r2)
print('================================================================================================\n')

## Run Random Forest Regressor with feature reduction

model6 = GradientBoostingRegressor()
model6.fit(Xfield_train2, yfield_train2)
yfield_pred_gbr = model6.predict(Xfield_test2)

print('\n')
print('========== Results Evaluation (Gradient Boosting Regressor without feature reduction) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield_test2, yfield_pred_gbr)))
r2 = r2_score(yfield_test2, yfield_pred_gbr)
print('R-Squared Score: ', r2)
print('================================================================================================\n')


## Run on all data based on the best performing model (model5)
yfield_pred_gbr_all = model5.predict(Xfield1)

print('\n')
print('========== Results Evaluation (Gradient Boosting Regressor All) ==========')
print('RMSE:', np.sqrt(mean_squared_error(yfield1, yfield_pred_gbr_all)))
r2 = r2_score(yfield1, yfield_pred_gbr_all)
print('R-Squared Score: ', r2)
print('==========================================================================\n')


print('Total preditions: ' ,len(yfield_pred_gbr_all))

df_field_attributes['Predicted Value (Euros) - Gradient Boosting'] = yfield_pred_gbr_all

df_field_attributes['Predicted Value (Euros) - Gradient Boosting'] = df_field_attributes['Predicted Value (Euros) - Gradient Boosting'].apply(formatInt)

print(df_field_attributes[:1000].filter(['Value (Euros)', 'Predicted Value (Euros) - Gradient Boosting']))
