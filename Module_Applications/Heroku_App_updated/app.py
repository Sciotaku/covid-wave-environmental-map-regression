import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import  scipy.signal.signaltools

# https://stackoverflow.com/questions/71106940/cannot-import-name-centered-from-scipy-signal-signaltools
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered

#import data from csv
df = pd.read_csv('https://raw.githubusercontent.com/difuse-dartmouth/21s_ENVS3/main/Module_Data/parish_data_v7.csv')
df1 = df.drop(columns=['FIPS', 'Parish Name'])
feature_names = list(df1.columns)

##Interactive Map variable definitions
#import data from geojson file
url2 = 'https://github.com/difuse-dartmouth/21s_ENVS3/blob/main/Module_Data/git_shapefiles/parish_data_v8.json?raw=true'
df2 = gpd.read_file(url2)
df2.sort_values(by=['FIPS'], inplace=True)
df2["center"] = df2.geometry.centroid
LA_parishes = df2[['parish_nam','geometry']].set_index("parish_nam")
map_variables = list(df2.columns)
# removed_cols = ['FIPS','OBJECTID','parish_nam', 'Shape_Leng','Shape_Area', 'geometry', 'center',
#                 'FIPS_1','parish_n_1']
removed_cols = ['FID','parish_nam', 'geometry', 'center',
                'FIPS_1','FIPS']
for item in removed_cols:
    map_variables.remove(item)
df_geoseries = gpd.GeoSeries(df2.geometry.centroid)

variable_list = ['First Wave of Covid Deaths per 10k',
 'Second Wave of Covid Deaths per 10k',
 'Third Wave of Covid Deaths per 10k',
 'Fourth Wave of Covid Deaths per 10k',
 'Covid Deaths per 10k thru 9/26/21']
for var in variable_list:
    df[var+" label"] = df["Parish Name"]+": "+df[var].round(2).astype("string")+" per 10k"

# Scale data to use for bubble map layer
data = df2[[m for m in map_variables]] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(2,50))
data_scaled = min_max_scaler.fit_transform(data)
scaled = pd.DataFrame(data_scaled, columns=feature_names)

#create variable for cancer alley parishes to add outline as additional spatial variable
cancer_alley_parishes = df2[df2["Cancer_All"] == 1]
cancer_alley_lats = []
cancer_alley_lons = []

for feature in cancer_alley_parishes.geometry.boundary:
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    else:
        continue
    for linestring in linestrings:
        x, y = linestring.xy
        cancer_alley_lats = np.append(cancer_alley_lats, y)
        cancer_alley_lons = np.append(cancer_alley_lons, x)
        cancer_alley_lats = np.append(cancer_alley_lats, None)
        cancer_alley_lons = np.append(cancer_alley_lons, None)

#dictionary for storing colorbar names and descriptive names for dropdown in app
color_dict = {'BlueYellow':'Viridis', 'PurpleOrange':'Inferno', 'Purples':'Purples', 'Reds':'Reds', 'RedBlue':'Rdbu',
                'BlueRed':'IceFire'}
color_scales = list(color_dict.keys())

##Correlation Matrix variable definitions
#create variable list and modified dataframes for dropdown and correlation matrix
covid_waves = feature_names[0:5]
variable_list1 = [var for var in feature_names if var not in covid_waves]

#create dataframes for each covid wave that exclude other covid waves for correlation matrix
First_Covid_Wave = df1.drop(columns=covid_waves[1:])
Second_Covid_Wave = df1.drop(columns=[covid_waves[0], covid_waves[2], covid_waves[3], covid_waves[4]])
Third_Covid_Wave = df1.drop(columns=[covid_waves[0], covid_waves[1], covid_waves[3], covid_waves[4]])
Fourth_Covid_Wave = df1.drop(columns=[covid_waves[0], covid_waves[1], covid_waves[2], covid_waves[4]])
Total_Current_Covid_Deaths = df1.drop(columns=covid_waves[:4])
feature_names2 = list(First_Covid_Wave.columns)

#use standard scaler to scale all the data before computing correlation coefficients
std_scaler = StandardScaler()
df_std = pd.DataFrame(std_scaler.fit_transform(First_Covid_Wave), columns=First_Covid_Wave.columns)
corr = df_std.corr()

#read in data from Git Repo
feature_names = list(df1.columns)

#list of color choices for Correlation Matrix
color_list = ['RedBlue','GreenPurple','RedGreyYellow','BrownGreen']

#create default figure that will appear when user opens tab in app
default_fig2 = px.imshow(corr,x=feature_names2,y=feature_names2,color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation Coefficient"))
default_fig2.update_xaxes(tickangle=45,tickfont_size=13)
default_fig2.update_coloraxes(colorbar_title_side="right",colorbar_title_font_size=15)
default_fig2.update_layout(coloraxis_colorbar=dict(thicknessmode="fraction", thickness=0.025,
                    lenmode="fraction", len=.9,xpad=100,x=0.7),margin=dict(t=10, b=0, l=50, r=0))
default_fig2.update_yaxes(tickfont_size=13)

##Linear Regression variable definitions
#create dataframe which retains name of parishes for hover label
df3 = df[['Parish Name']]
#options for turning regression line on and off
regr_list = ['Off','On']

#create default figure that will appear when user opens tab in app
default_fig3 = px.scatter(df1, x="Population", y="First Wave of Covid Deaths per 10k",template="simple_white")
default_fig3.update_layout(margin={"r":0,"t":10,"l":100,"b":50})
default_fig3.update_traces(marker_size=10,marker_color="black")
default_fig3.update_layout(hovermode='x')
default_fig3.update_xaxes(title="Population",showgrid=True,title_font_size=18,tickfont_size=15,automargin=True)
default_fig3.update_yaxes(title="First Wave of Covid Deaths per 10k",showgrid=True,title_font_size=18,tickfont_size=15,automargin=True)

# Function for drawing the interactive map
def draw_map(var, wave, color):
    fig = go.Figure()

    # Add choropleth layer
    fig.add_trace(go.Choroplethmapbox(geojson=LA_parishes.__geo_interface__, locations=df2.parish_nam,
                                      z=df[var], colorbar={'title': var},
                                      colorscale=color_dict.get(color), name=var
                                      ))
        # Add cancer alley borders
    fig.add_trace(go.Scattermapbox(mode = "lines", lat=cancer_alley_lats, lon=cancer_alley_lons, hoverinfo='skip',
                                   name = "Cancer Alley",marker=go.scattermapbox.Marker(color='DarkRed')))

    # Add bubble layer
    label = wave+" label"
    fig.add_trace(go.Scattermapbox(lat=df_geoseries.y, lon=df_geoseries.x, mode='markers',
                                   marker=go.scattermapbox.Marker(size=scaled[wave],color='Tomato'),
                                   text=df[label], hoverinfo="text", name=wave,
                                   ))

    fig.update_layout(mapbox_style="carto-positron",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      showlegend=True, mapbox_zoom=6,
                      mapbox_center={"lat": 30.9843, "lon": -91.9623},
                      margin={"r": 0, "t": 0, "l": 0, "b": 0}
                      )
#     fig.data[0].update(hovertemplate=df1['Parish Name'])
    return fig

#this is the default map before anything is selected from dropdown
default_fig1 = draw_map(variable_list1[0], covid_waves[0], color_scales[2])

#define Dash app, reference Dash Bootstrap as external stylsheet
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
#reference Dartmouth logo in Git repo for navbar brand
dartmouth_logo = "https://github.com/difuse-dartmouth/21s_ENVS3/raw/main/static/images/D-Pine_RGB.png?raw=true"
server = app.server

#define layout of app interface including navbar with title and brand and then the tabs for each graph
layout = html.Div([
    html.Div([
        html.Img(src=dartmouth_logo, height="40px")],style={'padding-left' : '5px','padding-top' : '-5px',
                                                            'width' : 'auto','display': 'inline-block'}),
    html.Div([
        html.P("COVID-19 Mortality in Louisiana")],
        style={'font-size' : '25px', 'font-weight' : 'bold',
               'padding-left': '10px','padding-top' : '5px','width' : 'auto',
               'display': 'inline-block'}),
    html.Div([
    dbc.Tabs(
                [
                    dbc.Tab(label="Interactive Map", tab_id="tab-1",tab_style={"margin-left": "10px"}),
                    dbc.Tab(label="Correlation Matrix", tab_id="tab-2"),
                    dbc.Tab(label="Linear Regression", tab_id="tab-3"),
                    dbc.Tab(label="Data Info", tab_id="tab-4"),
                    dbc.Tab(label="Credits", tab_id="tab-5")
                ],
                id="tabs",
                active_tab="tab-1",
            )],style={'padding-top': '5px','width' : 'auto'}),
    html.P(id="app-content", className="card-text")
        ])

app.layout = html.Div(
     [layout])

#define callbacks for interactivity with each tab
@app.callback(
    Output("app-content", "children"), [Input("tabs", "active_tab")]
)

#define tab content
def tab_content(active_tab):
    if active_tab == 'tab-1':
        #first tab is the interactive map with three dropdowns
        return html.Div([html.Div([
                html.Div([
                    html.Label(["Covid Wave:", dcc.Dropdown(
                        id='covid_wave', options=[{'label': i, 'value': i} for i in covid_waves],
                        value = covid_waves[0]
                    )],style={'width': '300px', 'display': 'inline-block'})],style={'width': '300px', 'display': 'inline-block'}
                ),
                html.Div([
                    html.Label(["Variable:", dcc.Dropdown(
                        id='variable', options=[{'label': i, 'value': i} for i in variable_list1],
                        value = variable_list1[0]
                    )],style={'width': '300px', 'display': 'inline-block'})],style={'width': '300px', 'display': 'inline-block'}
                ),
                html.Div([
                    html.Label(["Colorbar:", dcc.Dropdown(
                        id='colorbar', options=[{'label': i, 'value': i} for i in color_scales],
                        value = color_scales[2]
                    )],style={'width': '300px', 'display': 'inline-block'})],style={'width': '300px', 'display': 'inline-block'}
                ),
            ], style={
                'borderBottom': 'thin lightgrey solid', 'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '10px 5px'}),
            dcc.Graph(style={'height': '80vh'},id="graph4", figure=default_fig1)]),
    elif active_tab == 'tab-2':
        #second tab is the correlation matrix with two dropdowns
        return html.Div([html.Div([
            html.Div([
                html.Label(["Covid Wave:", dcc.Dropdown(
                    id='dropdown1',
                    options=[{'label': i, 'value': i} for i in variable_list],
                    value='First Wave of Covid Deaths per 10k'
                )],style={'width': '300px'})],style={'width': '300px', 'display': 'inline-block'},
                       ),
            html.Div([
                html.Label(["Colorbar:", dcc.Dropdown(
                    id='dropdown2',
                    options=[{'label': i, 'value': i} for i in color_list],
                    value='RedBlue'
                )],style={'width': '300px'})],style={'width': '300px', 'display': 'inline-block'},
                       )], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'}),
        dcc.Graph(config={'responsive': True},style={'height': '90vh','width' : '90vw'},id="graph1",figure=default_fig2)])
    elif active_tab == 'tab-3':
        #third tab is the linear regression plot with three dropdowns
        return html.Div([html.Div([
            html.Div([
                html.Label(["X Variable:", dcc.Dropdown(
                    id='x_variable',
                    options=[{'label': i, 'value': i} for i in feature_names],
                    value = 'Population'
                )],style={'width': '300px','display' : 'inline-block'})],style={'width': '300px', 'display': 'inline-block'}),
            html.Div([
                html.Label(["Y Variable:", dcc.Dropdown(
                    id='y_variable',
                    options=[{'label': i, 'value': i} for i in feature_names],
                    value = 'First Wave of Covid Deaths per 10k'
                 )],style={'width': '300px','display' : 'inline-block'})],style={'width': '300px', 'display': 'inline-block'}),
            html.Div([
                html.Label(["Regression:", dcc.Dropdown(
                    id='regression',
                    options=[{'label': i, 'value': i} for i in regr_list],
                    value = 'Off'
                 )],style={'width': '300px','display' : 'inline-block'})],style={'width': '300px', 'display': 'inline-block'})],style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'}),
        dcc.Graph(config={'responsive': True},style={'height': '80vh','width': '60vw'},id="graph3",figure=default_fig3)])
    elif active_tab == 'tab-4':
        #fourth tab provides info on all of the data included in the app and the sources
        return html.Div([
            html.Div([html.H5('Louisiana Dataset')],style={'padding-left': '10px','padding-top': '10px','font-weight': 'bold'}),
            html.Div([html.P('The dataset you will be working with includes 14 unique variables for'\
                             ' all 64 parishes (counties) of Louisiana. Below is a description of each variable'\
                             ' and the data sources where they were retrieved from:')],
                             style={'padding-left' : '10px','padding-right': '10px','font-size':'13pt'}),
            html.Div([
                    html.Ul([
                        dcc.Markdown('* **First Wave Covid Deaths per 10k:** Deaths per 10,000 people'\
                                         ' for the first COVID-19 wave, defined as from 1/22/2020 - 7/1/2020.<sup>1</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Second Wave Covid Deaths per 10k:** Deaths per 10,000 people for the'\
                                         ' second COVID-19 wave, defined as being from 7/1/2020 - 11/1/2020.<sup>1</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Third Wave Covid Deaths per 10k:** Deaths per 10,000 people'\
                                     ' for the third COVID-19 wave, defined as being from 11/1/2020 - 4/28/21.<sup>1</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Fourth Wave Covid Deaths per 10k:** Deaths per 10,000 people'\
                                     ' for the fourth COVID-19 wave, defined as being from 4/28/2021 - 9/26/21.<sup>1</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Covid Deaths per 10k thru 9/26/21:** Cumulative deaths per 10,000 people for'\
                                     ' the COVID-19 pandemic from 1/22/2020 - 9/26/21.<sup>1</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Population:** Population for the parish based upon 2020 Census.<sup>2</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Median Household Income:** Median household income.<sup>3</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Percent 65 and over:** Percentage of total county population that is'\
                                     ' 65 years of age and older.<sup>3</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Percent Black:** Percentage of population that is non-Hispanic Black or'\
                                     ' African American.<sup>3</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Poverty Rate:** Percentage of population living below the poverty level.'\
                                     ' The Census Bureau uses a set of money income thresholds that vary by family'\
                                     ' size and composition to determine who is in poverty. If a family\'s total income'\
                                     ' is less than the family\'s threshold, then that family and every individual in'\
                                     ' it is considered in poverty. The official poverty thresholds do not vary'\
                                     ' geographically, but they are updated for inflation using the Consumer Price'\
                                     ' Index (CPI-U). The official poverty definition uses money income before taxes'\
                                     ' and does not include capital gains or noncash benefits (such as public housing,'\
                                     ' Medicaid, and food stamps).<sup>4</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt','padding-right': '15px'}),
                        dcc.Markdown('* **Percent Rural:** Percentage of population living in a rural area.<sup>5</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Respiratory Hazard (RH) Weighted Ave:** The sum of hazard quotients for'\
                                     ' toxins that affect the respiratory system. Hazard quotients are defined as the'\
                                     ' ratio of the potential exposure to a substance and the level at which no adverse'\
                                     ' effects are expected (calculated as the exposure divided by the appropriate'\
                                     ' chronic or acute value). Because different air toxics can cause similar adverse'\
                                     ' health effects, combining hazard quotients from different toxics is often'\
                                     ' appropriate. A hazard index (HI) of 1 or lower means air toxics are unlikely'\
                                     ' to cause adverse noncancer health effects over a lifetime of exposure. The RH'\
                                     ' data is provided by census tract, so an average for the parish is computed as the'\
                                     ' weighted average by the population for the entire parish.<sup>6</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt','padding-right': '15px'}),
                        dcc.Markdown('* **TRI Total On Site Air Release (pounds):** The amount of toxic emissions'\
                                     ' released on site at all facilities in pounds within a parish for'\
                                     ' the entire year of 2019.<sup>7</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'}),
                        dcc.Markdown('* **Cancer Alley:** Whether parish is a part of “Cancer Alley” in Louisiana.'\
                                     ' Variable is binary (meaning it is either a 0 or 1 value), where 1 ='\
                                     ' Cancer Alley Parish.<sup>8</sup>',
                                     dangerously_allow_html=True,
                                    style={'font-size': '13pt'})
                    ],style={'padding-left': '25px'}),
                html.Div([
                    dcc.Markdown('<sup>1</sup>Retrieved from usafacts.org, data from the CDC and Louisiana'\
                                 ' Department of Health.',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>2</sup>Retrieved from usafacts.org, data from 2020 US census.',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>3</sup>Retrieved from the 2020 County Health Rankings, data from 2018 census.',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>4</sup>Data from the 2020 Census.',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>5</sup>Retrieved from the 2020 County Health Rankings, data from 2010 census.',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>6</sup>Data from the 2014 National Air Toxics Assessment (EPA).',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>7</sup>Data is from the 2019 Toxics Release Inventory (TRI) and is summed for'\
                                 ' all facilities within a parish.',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                    dcc.Markdown('<sup>8</sup>Based on the definition of Cancer Alley in James et al. 2012'\
                                 ' (https://www.mdpi.com/1660-4601/9/12/4365/htm).',
                                     dangerously_allow_html=True,
                                    style={'padding-left': '10px','font-size': '13pt'}),
                ])
            ])
        ])
    elif active_tab == 'tab-5':
        #fifth tab provides acknowledgements for DIFUSE and ENVS3 teams
        return html.Div([
            dcc.Markdown('<u>Attribution Note:</u> <i>please acknowledge the ENVS3 Team and the Dartmouth DIFUSE program if'\
            ' you share or utilize this resource</i>',dangerously_allow_html=True,
                         style={'padding-top': '20px','padding-left': '10px','font-size': '13pt'}),
            dcc.Markdown('<b>ENVS3 Team:</b> James Busch (Ph.D. Candidate), William Chen (\'23), J.T. Erbaugh'\
                         ' (NSF Postdoctoral Fellow), Richard Howarth (Professor of Environmental Studies)',
                         dangerously_allow_html=True,
                         style={'padding-top': '20px','padding-left': '10px','font-size': '13pt'}),
            dcc.Markdown('<b>DIFUSE Project Managers:</b> Tiffany Yu (\'21) and Taylor Hickey (\'23)',
                         dangerously_allow_html=True,
                         style={'padding-top': '20px','padding-left': '10px','font-size': '13pt'}),
            dcc.Markdown('<b>DIFUSE PI\'s:</b> Prof. Petra Bonfert-Taylor (Thayer School), Prof. Laura Ray (Thayer School),'\
                         ' Prof. Scott Pauls (Mathematics), Prof. Lori Loeb (Computer Science)',
                         dangerously_allow_html=True,
                         style={'padding-top': '20px','padding-left': '10px','font-size': '13pt'}),
            dcc.Markdown('<b>Development Team:</b> James Busch compiled and authored code for the web application, correlation matrix,'\
                         ' and linear regression components. William Chen compiled and authored the code used for the interactive map.',
                         dangerously_allow_html=True,
                         style={'padding-top': '20px','padding-left': '10px','font-size': '13pt'}),
            html.Div(['The DIFUSE project is supported by the National Science Foundation under grant no. DUE- 1917002'],
                   style={'padding-left': '10px','font-size': '13pt','font-style' : 'italic'})
        ])

#define callbacks for dropdowns in interactive map
@app.callback(
    dash.dependencies.Output("graph4", "figure"),
    [dash.dependencies.Input('variable', 'value'), dash.dependencies.Input('covid_wave', 'value'),
     dash.dependencies.Input('colorbar', 'value')])
#function that updates the map based on dropdown choice
def update_graph(variable, covid_wave, colorbar):
    return draw_map(variable, covid_wave, colorbar)

#define callbacks for dropdowns in correlation matrix
@app.callback(
    dash.dependencies.Output("graph1", "figure"),
     [dash.dependencies.Input('dropdown1', 'value'),
     dash.dependencies.Input('dropdown2', 'value')])
#function that updates correlation matrix based on dropdown choice
def filter_heatmap(dropdown1,dropdown2):
    if dropdown1 == 'First Wave of Covid Deaths per 10k':
        dropdown1 = First_Covid_Wave
    elif dropdown1 == 'Second Wave of Covid Deaths per 10k':
        dropdown1 = Second_Covid_Wave
    elif dropdown1 == 'Third Wave of Covid Deaths per 10k':
        dropdown1 = Third_Covid_Wave
    elif dropdown1 == 'Fourth Wave of Covid Deaths per 10k':
        dropdown1 = Fourth_Covid_Wave
    elif dropdown1 == 'Covid Deaths per 10k thru 9/26/21':
        dropdown1 = Total_Current_Covid_Deaths
    if dropdown2 == 'RedBlue':
        dropdown2 = 'RdBu_r'
    elif dropdown2 == 'GreenPurple':
        dropdown2 = 'curl'
    elif dropdown2 == 'RedGreyYellow':
        dropdown2 = 'oxy'
    else:
        dropdown2 = 'BrBG'
    feature_names = list(dropdown1.columns)
    std_scaler = StandardScaler()
    df_std = pd.DataFrame(std_scaler.fit_transform(dropdown1), columns=dropdown1.columns)
    corr = df_std.corr()
    fig = px.imshow(corr,x=feature_names,y=feature_names,color_continuous_scale=dropdown2,
                labels=dict(color="Correlation Coefficient"))
    fig.update_xaxes(tickangle=45,tickfont_size=13)
    fig.update_yaxes(tickfont_size=13)
    fig.update_coloraxes(colorbar_title_side="right",colorbar_title_font_size=15)
    fig.update_layout(coloraxis_colorbar=dict(thicknessmode="fraction", thickness=0.025,
                    lenmode="fraction", len=.9,xpad=100,x=0.7),margin=dict(t=10, b=0, l=50, r=0))
    return fig

#define callbacks for dropdowns in regression panel
@app.callback(
    dash.dependencies.Output("graph3", "figure"),
     [dash.dependencies.Input('x_variable', 'value'),
     dash.dependencies.Input('y_variable', 'value'),
     dash.dependencies.Input('regression', 'value')])
#function that updates regression plot based on dropdown choice
def update_graph(x_variable, y_variable,regression):
    if regression == 'Off':
        if x_variable == y_variable:
            fig = go.Figure()
            fig.update_layout()
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.add_annotation(text="X Variable cannot equal Y Variable",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28))
            return fig
        else:
            df2 = df1[[x_variable,y_variable]]
            df2 = df2.dropna(axis='rows')
            fig = px.scatter(x=df2[x_variable],y=df2[y_variable], template="simple_white")
            fig.update_layout(margin={"r":0,"t":10,"l":100,"b":50})
            fig.update_traces(marker_size=10,marker_color="black")
            fig.update_xaxes(title=x_variable,showgrid=True,title_font_size=18,tickfont_size=15,automargin=True)
            fig.update_yaxes(title=y_variable,showgrid=True,title_font_size=18,tickfont_size=15,automargin=True)
            fig.data[0].update(hovertemplate=df3['Parish Name'])
    else:
        if x_variable == y_variable:
            fig = go.Figure()
            fig.update_layout()
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.add_annotation(text="X Variable cannot equal Y Variable",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28))
            return fig
        else:
            df2 = df1[[x_variable,y_variable]]
            df2 = df2.dropna(axis='rows')
            regr = LinearRegression()
            x = df2[[x_variable]].values
            y=df2[[y_variable]].values
            regr.fit(x,y)
            y_pred = regr.predict(x)
            y_inter = regr.intercept_.item()
            if np.sign(y_inter) == -1:
                sign = '-'
            else:
                sign = '+'
            slope = regr.coef_.item()
            r2 = r2_score(y, y_pred)
            r2_eq = 'R\N{SUPERSCRIPT TWO}= %.2f' % r2 + '<br>'
            eq='y = %.3f' % slope + 'x ' + sign + ' ' '%.3f' % y_inter
            fig = px.scatter(x=df2[x_variable],y=df2[y_variable],template="simple_white",trendline='ols',
                     trendline_color_override='red')
            fig.update_layout(margin={"r":0,"t":10,"l":100,"b":50})
            fig.update_traces(marker_size=10,marker_color="black")
            fig.update_xaxes(title=x_variable,showgrid=True,title_font_size=18,tickfont_size=15,automargin=True)
            fig.update_yaxes(title=y_variable,showgrid=True,title_font_size=18,tickfont_size=15,automargin=True)
            fig.data[1].update(hovertemplate=r2_eq+eq)
            fig.data[0].update(hovertemplate=df3['Parish Name'])
            std_scaler = StandardScaler()
            df_std = pd.DataFrame(std_scaler.fit_transform(df1), columns=df1.columns)
            X1 = df2[[x_variable]].values
            Y1 = df2[[y_variable]].values
            x1 = X1.flatten()
            y1 = Y1.flatten()
            data = pd.DataFrame({'x': x1, 'y': y1})
            model = ols("y ~ x", data).fit()
            p_value = model.pvalues[1]
            if p_value > 0.01:
                p_value2 = ('P > |t| = %.4f' % p_value)
            else:
                p_value2 = ('P > |t| = <0.01')
            fig.add_annotation(xref='paper',yref='paper', x=0.001, y=0.999,
                text=p_value2,showarrow=False,font=dict(size=18))
            fig.add_annotation(xref='paper',yref='paper', x=0.99, y=0.96,
                text=r2_eq,showarrow=False,font=dict(size=18))
            fig.add_annotation(xref='paper',yref='paper', x=0.99, y=0.999,
                text=eq,showarrow=False,font=dict(size=18))
    return fig

#run the app on the defined server
if __name__ == "__main__":
    app.run_server(debug=False)
