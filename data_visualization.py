"""
Author: William Gemba

This is a file to perform analysis and visualization of the cleaned FIFA 18 Player Data Set.

Python 3.8.6
"""
import os
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools
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

os.chdir('<Insert PATH Here>')
data = pd.read_csv('FIFA18playerdata_CLEANED.csv')

### View Data ###
data.head()
data.info()

"""                         Visualizations                           """

dframe_nations = data.groupby(by='Nationality').size().reset_index()
dframe_nations.columns = ['Nation', 'Count']

print(dframe_nations)

### Top 10 Nations By Player Count ###
print(dframe_nations.sort_values(by='Count', ascending = False).reset_index(drop=True).head(10))

### Visualize Nationalities by Frequency on a Map Projection ###
## Must combine all U.K. countries into one for visualizaiton purposes ##
print(dframe_nations[(dframe_nations['Nation'] == 'England') | (dframe_nations['Nation'] == 'Wales') |
               (dframe_nations['Nation'] == 'Scotland') | (dframe_nations['Nation'] == 'Northern Ireland')])

tempdataframe = pd.DataFrame(data= [['United Kingdom', 2148]], columns=['Nation', 'Count'])
dframe_nations = dframe_nations.append(tempdataframe, ignore_index=True)

trace1 = dict(type='choropleth', locations = dframe_nations['Nation'], z = dframe_nations['Count'],
             locationmode = 'country names', colorscale = 'reds')
map_layout = go.Layout(title = '<b>Number of Players From Each Nation</b>',
                       geo = dict(showocean = True, oceancolor = '#AEDFDF', projection = dict(type = 'mercator')))

fig_map = go.Figure(data=[trace1], layout=map_layout)
py.iplot(fig_map)

### Visualize Age Distribution ###
trace2 = go.Histogram(x=data['Age'], nbinsx=55, opacity=0.7)

layout = go.Layout(title='<b>Players Age Distribution<b>',
                   xaxis=dict(title='<b><i>Age</b></i>'),
                   yaxis=dict(title='<b><i>Count</b></i>'),
                  )

fig_age = go.Figure(data=[trace2], layout=layout)
py.iplot(fig_age)

### Visualize Height and Weight Distributions ###

fig_hw = tools.make_subplots(rows=1, cols=2)

trace3a = go.Histogram(x=data['Height (cm)'], nbinsx=25, opacity=0.7, name='Height (cm)')
trace3b = go.Histogram(x=data['Weight (lbs)'], nbinsx=30, opacity=0.7, name='Weight (lbs')

fig_hw.append_trace(trace3a, 1,1)
fig_hw.append_trace(trace3b, 1,2)

fig_hw['layout'].update(title= '<b>Height & Weight Distribution</b>', \
                        xaxis=dict(automargin=True),
                        yaxis=dict(title='<b><i>Count</i></b>'))

py.iplot(fig_hw)

### Visualize Position Distribution ###

trace4 = go.Pie(values=data['Position'].value_counts().values,
                labels=data['Position'].value_counts().index.values,
                hole=0.3
                )

layout = go.Layout(title='<b>Distribution of Players by Exact Position</b>')

fig_pie1 = go.Figure(data=[trace4], layout=layout)
py.iplot(fig_pie1)

### Same thing but for Position Groupings ###
trace4a = go.Pie(values=data['Position Grouping'].value_counts().values,
                labels=data['Position Grouping'].value_counts().index.values,
                hole=0.3
                )

layout = go.Layout(title='<b>Distribution of Players by Position Grouping</b>')

fig_pie2 = go.Figure(data=[trace4a], layout=layout)
py.iplot(fig_pie2)

### Preferred Foot Distribution ###

trace5 = go.Pie(values=data['Preferred Foot'].value_counts().values, labels=data['Preferred Foot'].value_counts().index.values, hole=0.3)
layout = go.Layout(title='<b>Preferred Foot Distribution</b>')

fig_foot = go.Figure(data=[trace5], layout=layout)

py.iplot(fig_foot)

### Relationship Between Work Rate and Overall Rating ###
trace6 = go.Violin(x=data['Work Rate'], y=data['Overall'])

layout = go.Layout(title='<b>Relationship Between Work Rate and Overall Rating</b>', xaxis=dict(title='<b><i>Work Rate</b></i>'), yaxis=dict(title='<b><i>Overall</b></i>'))

fig_wrovr = go.Figure(data=[trace6], layout=layout)

py.iplot(fig_wrovr)
