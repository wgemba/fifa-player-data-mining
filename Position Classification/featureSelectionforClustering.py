"""
Author: William Gemba

This is a file to perform Data Mining Tasks of the cleaned FIFA 18 Player Data Set.

Python 3.8.6
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode
import plotly.io as pio
import cufflinks as cf
import warnings

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

# Define function to convert previously greated 'Position Grouping' class into numeric variable
def pos_numerize(val):
    if val =='GK':
        return 0
    elif val == 'DEF':
        return 1
    elif val == 'MID':
        return 2
    else:
        return 3

os.chdir('<Insert PATH Here>')

data = pd.read_csv('FIFA18playerdata_CLEANED.csv', index_col='Name')

data.head()
data.info()

data = data.drop(columns = ['Jersey Number', 'Position', 'International Reputation', 'GKDiving', 'GKHandling',
                              'GKKicking', 'GKPositioning', 'GKReflexes', 'Nationality', 'Club', 'Work Rate', 'Body Type', 'Weak Foot',
                            'Preferred Foot'], inplace=False)


index_names_GK = data[data['Position Grouping'] == 'GK'].index ## Drop GK for test
data = data.drop(index_names_GK, inplace = False)

data['FWD'] = (data['RF'] + data['ST'] + data['LF'] + data['RS'] + data['LS'] + data['CF']) / 6

data['MID'] = (data['LW'] + data['RCM'] + data['LCM'] + data['LDM'] + data['CAM'] + data['CDM'] + \
                data['RM'] + data['LAM'] + data['LM'] + data['RDM'] + data['RW'] + data['CM'] + data['RAM'])\
                /13

data['DEF'] = (data['RCB'] + data['CB'] + data['LCB'] + data['LB'] + data['RB'] + data['RWB']\
                 + data['LWB']) / 7

data.drop(columns=['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB', 'LDM', 'CAM', 'CDM',
                     'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM', 'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB'], inplace=True)


data['Position Grouping'] = data['Position Grouping'].apply(pos_numerize)


data['Speed'] = data[['Acceleration', 'SprintSpeed']].mean(axis=1)

data['Shooting'] = data[['Finishing', 'LongShots', 'Penalties', 'Positioning', 'ShotPower', 'Volleys']].mean(axis=1)

data['Passing'] = data[['Crossing', 'Curve', 'FKAccuracy', 'LongPassing', 'ShortPassing', 'Vision']].mean(axis=1)

data['DribblingStats'] = data[['Agility', 'Balance', 'BallControl', 'Composure', 'Dribbling', 'Reactions']].mean(axis=1)

data['DefensiveStats'] = data[['HeadingAccuracy', 'Interceptions', 'Marking', 'SlidingTackle', 'StandingTackle']].mean(axis=1)

data['Physical'] = data[['Aggression', 'Jumping', 'Stamina', 'Strength']].mean(axis=1)

data = data.drop(columns=['Acceleration', 'SprintSpeed', 'Finishing', 'LongShots', 'Penalties', 'Positioning', 'ShotPower', 'Volleys',
                          'Crossing', 'Curve', 'FKAccuracy', 'LongPassing', 'ShortPassing', 'Vision', 'Agility', 'Balance', 'BallControl', 'Composure', 'Dribbling', 'Reactions',
                          'HeadingAccuracy', 'Interceptions', 'Marking', 'SlidingTackle', 'StandingTackle', 'Aggression', 'Jumping', 'Stamina', 'Strength'])

data.head()
data.info()

print(data[:3])

### Correlation Matrix - Made into a comment after no longer needed to run ###
corr_matrix = data.corr().abs()
print(corr_matrix)
print('\n')
print(corr_matrix['Position Grouping'], )

plt.figure(figsize=(20,10))
ax = sns.heatmap(corr_matrix)
plt.show()

# Chose highest correllating features to Position Grouping
feat_redu_data = data.filter(['Finishing', 'Volleys', 'Dribbling', 'ShotPower', 'LongShots', 'Interceptions', 'Positioning', 'Vision',
'Penalties', 'Marking', 'StandingTackle', 'SlidingTackle', 'Position Grouping'], axis=1)

print(feat_redu_data[:5])

corr_matrix2 = feat_redu_data.corr().abs()

plt.figure(figsize=(20,10))
ax = sns.heatmap(corr_matrix2, annot=True)
plt.show()

print('Absolute Correlation to Position Grouping')
print(corr_matrix2)
print('\n')
print(corr_matrix2['Position Grouping'], )

# Export to CSV
#feat_redu_data.to_csv(r'<Insert Path Here>/FIFA18playerdata_CLEANED_featurereduc.csv', index = True)

