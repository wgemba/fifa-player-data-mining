"""
Author: William Gemba

This is a file to perform Data Mining Tasks of the cleaned FIFA 18 Player Data Set.

"""
import os
from plotly.offline import init_notebook_mode
import plotly.io as pio
import cufflinks as cf
import warnings
import pandas as pd

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
data = pd.read_csv('FIFA18playerdata_CLEANED.csv')

data.head()
data.info()

data1 = data.copy()
data1.head()
data1.info()

data1 = data1.drop(columns = ['Jersey Number', 'Position', 'International Reputation', 'GKDiving', 'GKHandling',
                              'GKKicking', 'GKPositioning', 'GKReflexes', 'Nationality'], inplace=False)
print('Data without useless info: \n')
data1.head()
data1.info()

index_names_GK = data1[data1['Position Grouping'] == 'GK'].index ## Drop GK for test
data1 = data1.drop(index_names_GK, inplace = False)

data1.head()
data1.info()

data1['Speed'] = data1[['Acceleration', 'SprintSpeed']].mean(axis=1)

data1['Shooting'] = data1[['Finishing', 'LongShots', 'Penalties', 'Positioning', 'ShotPower', 'Volleys']].mean(axis=1)

data1['Passing'] = data1[['Crossing', 'Curve', 'FKAccuracy', 'LongPassing', 'ShortPassing', 'Vision']].mean(axis=1)

data1['DribblingStats'] = data1[['Agility', 'Balance', 'BallControl', 'Composure', 'Dribbling', 'Reactions']].mean(axis=1)

data1['DefensiveStats'] = data1[['HeadingAccuracy', 'Interceptions', 'Marking', 'SlidingTackle', 'StandingTackle']].mean(axis=1)

data1['Physical'] = data1[['Aggression', 'Jumping', 'Stamina', 'Strength']].mean(axis=1)

data1.head()
data1.info()

print(data1[:3])

data1 = data1.sort_values('Overall', ascending=False)
print('\n')
print('Data Frame sorted by Overall in descending order: \n')
print(data1[:3])

topplayers = data1[data1['Overall'] >= 85]
print('\nResulting Data Frame: \n', topplayers)

### Export to CSV
#topplayers.to_csv(r'<Insert PATH Here', index = False)

#data1.drop(columns=['Position Grouping'], inplace=True)
