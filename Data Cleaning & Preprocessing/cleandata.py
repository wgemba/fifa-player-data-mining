# -*- coding: utf-8 -*-
"""
Author: William Gemba

This is a script file to clean the FIFA 18 Player Data Set.

"""
import os
import pandas as pd
from plotly.offline import init_notebook_mode
import cufflinks as cf
import warnings
warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)
cf.go_offline()

### Set Display Options for Panda ###
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

os.chdir('C:/Users/willi/Documents/Data Science Practice Projects/Fordham Projects/Data_Mining_Final_Project_Clustering_Classification/')
data = pd.read_csv('FIFA18playerdata.csv')

""" View data"""
data.head()
data.info()

""" Identify Number of Categorical and Numerical Features"""
print('Number of Categorical Columns: ', len(data.select_dtypes(include=object) .columns))
print('Number of Numerical Columns: ', len(data.select_dtypes(include=object) .columns))

"""                             DATA FILE CLEANING & PREPROCESSING BEGINS HERE                                       """
### Remove Unnecassary columns ###
data.drop(columns=['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club Logo', 'Special', 'Real Face', 'Release Clause', 'Joined', 'Contract Valid Until', 'Loaned From'], inplace=True)

data.head()
data.info()

### Display Total Empty Cell Numbers for Each Attribute ###kiiii
print(data.isnull().sum())

### Replace empty club values with 'No Club' ###
data['Club'].fillna(value='No Club', inplace=True)

### Look at the 'Preferred Foot' Empty records ###
print(data[data['Preferred Foot'].isna()].head())

data.drop(index=data[data['Preferred Foot'].isna()].index, inplace=True) # Too many N/A values

data.head()
data.info()

print(data.isnull().sum())

### Look at 'Position' empty records ###
print(data[data['Position'].isna()][['Name', 'Nationality', 'LS', 'ST','RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                              'LAM', 'CAM', 'RAM', 'LM', 'LCM','CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM',
                              'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head())

data.drop(index=data[data['Position'].isna()].index, inplace=True)

data.head()
data.info()

print(data.isnull().sum())

### Remaining nulls are position stats for goalkeepers; cannot remove therefore fill as '0' ###
data.fillna(value=0, inplace=True)

data.head()
data.info()

print(data.isnull().sum().sum())

### Convert Categorical features to Numerical Features ###
print(data.select_dtypes(include=object).columns) # Shows all the non-numeric features

### Define Function to Convert Discrete Numerical Values to Continuous Ones ###
def convert_Currency(val):
    if val[-1] == 'M':
        val = val[1:-1]
        val = float(val) * 1000000
        return val
    elif val[-1] == 'K':
        val = val[1:-1]
        val = float(val) * 1000
        return val
    else:
        return 0
### Apply Converting Function to 'Value', 'Weight' Attributes ###
data['Value (Euros)'] = data['Value'].apply(convert_Currency)
data['Weekly Wage (Euros)'] = data['Wage'].apply(convert_Currency)

data.drop(columns=['Value', 'Wage'], inplace=True)

print(data.head())

### Define Height and Weight Conversion Functions
def splitHeight(val):
    ft = val.split("'")[0]
    ich = val.split("'")[1]
    ht = (int(ft) * 30.48) + (int(ich) * 2.54) # 1 Ft = 30.48 cm and 1 Inch = 2.54 cm
    return ht

def splitWeight(val):
    weight = int(val.split('lbs')[0])
    return weight

### Convert 'Height' and 'Weight' to Numeric Values ###
data['Height (cm)'] = data['Height'].apply(splitHeight)
data['Weight (lbs)'] = data['Weight'].apply(splitWeight)

data.drop(columns=['Height', 'Weight'], inplace=True)

print(data.head())

### Define Skill Conversion Function ###
def convert_Skill(val):
    if type(val) == str:
        s1 = val[0:2]
        s2 = val[-1]
        val = int(s1) + int(s2)
        return val
    else:
        return val

### Convert All Skill Attributes to Numeric Values ###
skill_attr = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

for attribute in skill_attr:
    data[attribute] = data[attribute].apply(convert_Skill)

print(data[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head())

print('The remaining non-numeric attributes are: ')
print(data.select_dtypes(include=object).columns)

### Fix 'Work Rate', 'Body Type', and 'Position' Attributes ###

print(data['Work Rate'].unique())

print(data['Body Type'].unique())

print(data['Position'].unique())

# Fix 'Body Type' columns #
data['Body Type'][data['Body Type'] == 'Messi'] = 'Lean'
data['Body Type'][data['Body Type'] == 'C. Ronaldo'] = 'Normal'
data['Body Type'][data['Body Type'] == 'Neymar'] = 'Lean'
data['Body Type'][data['Body Type'] == 'Courtois'] = 'Lean'
data['Body Type'][data['Body Type'] == 'PLAYER_BODY_TYPE_25'] = 'Normal' #PLAYER_BODY_TYPE_25 is for Mohammed Salah  = Normal body type.
data['Body Type'][data['Body Type'] == 'Shaqiri'] = 'Stocky'
data['Body Type'][data['Body Type'] == 'Akinfenwa'] = 'Stocky'

print(data['Body Type'].unique())

## Categorize Positions ##
print(data['Position'].unique())
print(data['Position'].nunique())

def categorize_Pos(val):
    if val == 'RF' or val == 'CF' or val == 'LF' or val == 'ST' or val == "LS" or val == "RS" :
        val = "FWD"
        return val

    elif val == 'LW' or val == 'LM' or val == 'LDM' or val == 'LCM' or val == 'LAM' or val == 'RW' or val == 'RM' or \
            val == 'RDM' or val == 'RAM' or val == 'RCM' or val == 'CM' or val == 'CDM' or val == 'CAM' :
        val = "MID"
        return val
    elif val == 'RB' or val == 'RCB' or val == 'RWB' or val == 'LB' or val == "LCB" or val == "LWB" or val == 'CB' :
        val = "DEF"
        return val
    else:
        return val

data['Position Grouping'] = data['Position'].apply(categorize_Pos)

print(data['Position Grouping'].value_counts())

data.values = data.values.str.replace(' ', '')

"""   END OF DATA CLEANING   """

### Export to CSV ###

#data.to_csv(r'<Insert Path Here>/FIFA18playerdata_CLEANED.csv', index = False)