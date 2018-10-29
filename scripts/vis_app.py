# -*- coding: utf-8 -*-
"""
Visualize the ensemble streamflow predictions

@author: Travis Williams
"""
import copy
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from flask_cache import Cache
import glob
import json
import numpy as np
import os
import pandas as pd
import random
import scipy
os.chdir("c:\\users\\user\\github\\ESP_simulation")
# In[] Set up application and server
app = dash.Dash(__name__)

# The stylesheet is based one of the DASH examples
# (oil and gas extraction in New York)
app.css.append_css({'external_url': 'https://rawgit.com/WilliamsTravis/' +
                    'PRF-USDM/master/dash-stylesheet.css'})

# Create server object
server = app.server

# Create and initialize a cache for storing data - data pocket
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(server)

# Create a container for the graphs
layout = dict(
    autosize=True,
    height=500,
    font=dict(color='black'),
    titlefont=dict(color='black',
                   size='20',
                   weight='bold'),
    margin=dict(
        l=35,
        r=35,
        b=65,
        t=55,
        pad=4
    ),
    # hovermode="closest",
    plot_bgcolor="white",
    paper_bgcolor="lightblue",
    legend=dict(font=dict(size=10), orientation='h'))

# In[] Get data -2 sets
files = glob.glob("data\\*")
files = [f for f in files if "historical" not in f]
dolc_hist = pd.read_csv("data\\DOLC2_historical.csv")
mphc_hist = pd.read_csv("data\\MPHC2_historical.csv")

dolc_files = [f for f in files if "DOLC2" in f]
mphc_files = [f for f in files if "MPHC2" in f]

dolc_dfs = {f[-8:-4]: pd.read_csv(f) for f in dolc_files}
yrs = [f[-8:-4] for f in dolc_files]
year_options = [{'label': y, 'value': y} for y in yrs]

mphc_dfs = {f[-8:-4]: pd.read_csv(f) for f in mphc_files}

df_dict = {'MPHC2': mphc_dfs,
           'DOLC2': dolc_dfs}

hist_dict = {'MPHC2': mphc_hist,
             'DOLC2': dolc_hist}

site_options = [{'label': "McPhee Reservoir", 'value': 'MPHC2'},
                {'label': "McPhee Reservoir Entry Point at Dolores",
                 'value': 'DOLC2'}]

# In[] Set up HTML structure

app.layout = html.Div([

        html.H2("Ensemble Streamflow Predictions at the McPhee Reservoir",
                style={'text-align': 'center'}),

        html.Div(className="row",
                 children=[html.Div([dcc.Dropdown(id='site_choice',
                                                  options=site_options,
                                                  placeholder="McPhee Reservoir")],
                                    className="two columns"),

                           html.Div([dcc.Dropdown(id="year",
                                                  options=year_options,
                                                  value='2018')],
                                    className="one column",
                                    style={'width': '90'})]),

        html.Div(className="row",
                 children=[html.Div(children=[dcc.Graph(id='cbrfc_graph')],
                                    className="six columns"),

                           html.Div(children=[dcc.Graph(id='cbrfc_history')],
                                    style={'float': 'right'},
                                    className='six columns')]),

        html.Div(className="row",
                 children=[html.Div(children=[dcc.Graph(id='err_evolve')],
                                    className='six columns'),
                           html.Div(children=[dcc.Graph(id='uncrtnty_evolve')],
                                    className='six columns')]),

        html.Div(className="row",
                 children=[html.Div(children=[dcc.Graph(id='err_evolve_all')],
                                    className='six columns'),
                           html.Div(children=[dcc.Graph(id='uncrtnty_evolve_all')],
                                    className='six columns')]),

        html.Br(),

        html.Div(className="twelve columns",
                 children=[dcc.Graph(id='our_graph')]),

        html.Div(className="row",
                 style={'width': '100%',
                        'margin-bottom': '75'},
                 children=[
                         html.Div(
                             className="three columns",
                             style={
                                    'height': '200',
                                    'margin-right': '10',
                                    'margin-left': '150'},
                             children=[
                                     html.P('S.D.'),
                                     html.P(id='sd_output'),
                                     dcc.Slider(id='sd',
                                                min=0,
                                                max=25,
                                                step=1,
                                                value=0,
                                                updatemode='drag',
                                                vertical=True,
                                                marks={0: {'label': '0'},
                                                       25: {'label': '25'}})]),

                         html.Div(
                             className="three columns",
                             style={'height': '200',
                                    'margin-right': '10'},
                             children=[
                                     html.P('S.D. 2'),
                                     html.P(id='sd_output2'),
                                     dcc.Slider(id='sd2',
                                                min=0,
                                                max=25,
                                                step=1,
                                                value=0,
                                                updatemode='drag',
                                                vertical=True,
                                                marks={0: {'label': '0'},
                                                       25: {'label': '25'}})])
                       ]),
        html.Hr(),
        ])  # *END


# In[]
@app.callback(Output('cbrfc_graph', 'figure'),
              [Input('year', 'value'),
               Input('site_choice', 'value')])
def makeGraph(year, site_choice):
    if not site_choice:
        site_choice = 'MPHC2'
    dfs = df_dict[site_choice]
    df = dfs[year]
    df = df[75:295] # January to August
    obs = df['Observed Accumulation'].dropna()
    final = obs.iloc[-1]
    var = round(np.nanvar(df['ESP 50']), 2)
    yaxis = dict(range=[0, 700])
    df['ratio10'] = df['ESP 10'].apply(lambda x: str(round(x/final*100, 2)) + "%")
    df['text10'] = df['ESP 10'].astype(str) + " KAF; " + df['ratio10']
    df['ratio50'] = df['ESP 50'].apply(lambda x: str(round(x/final*100, 2)) + "%")
    df['text50'] = df['ESP 50'].astype(str) + " KAF; " + df['ratio50']
    df['ratio90'] = df['ESP 90'].apply(lambda x: str(round(x/final*100, 2)) + "%")
    df['text90'] = df['ESP 90'].astype(str) + " KAF; " + df['ratio90']

    annotation = dict(
        text="<b>Forecast Variance: " + "{:,}".format(var) + "</b>",
        x=year + '-06-25',
        y=650,
        font=dict(size = 17),
        showarrow=False)
    
    data = [dict(type='line',
                 line=dict(color='#8ad88d',
                           width=2,
                           dash="dashdot"),
                 x=df.Date,
                 y=df['ESP 90'],
                 name='p90',
                 text=df['text90'],
                 hoverinfo='text'),
            dict(type='line',
                 line=dict(color='#04a00a',
                           width=4),
                 x=df.Date,
                 y=df['ESP 50'],
                 name="p50",
                 text=df['text50'],
                 hoverinfo='text'
                 ),
            dict(type='line',
                 line=dict(color='#8ad88d',
                           width=2,
                           dash="dashdot"),
                 x=df.Date,
                 y=df['ESP 10'],
                 name='p10',
                 text=df['text10'],
                 hoverinfo='text'),
            dict(type='line',
                 line=dict(color='blue',
                           width=4),
                 x=df.Date,
                 y=df['Observed Accumulation'],
                 name="Observation (KAF)")]

    layout_c = copy.deepcopy(layout)
    layout_c['title'] = ("Colorado Basin River Forecast Center" +
                         "'s " + '"ESP"' + " - " + site_choice + " " + year)
    layout_c['dragmode'] = 'select'
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['annotations'] = [annotation]
    layout_c['yaxis'] = yaxis
    layout_c['showlegend'] = True
    figure = dict(data=data, layout=layout_c)

    return figure


@app.callback(Output('cbrfc_history', 'figure'),
              [Input('site_choice', 'value')])
def makeGraph2(site_choice):
    if not site_choice:
        site_choice = 'MPHC2'
    df = hist_dict[site_choice]
    var = round(np.nanvar(df['vol']), 2)
    yaxis = dict(range=[0, 700])
    df['text'] = df['vol'].astype(str) + "KAF"
    annotation = dict(
        text="<b>Streamflow Variance: " + "{:,}".format(var) + "</b>",
        x=2010,
        y=650,
        font=dict(size=17),
        showarrow=False)

    data = [dict(type='line',
                 line=dict(color='blue',
                           width=4),
                 x=df['year'],
                 y=df['vol'],
                 text=df['text'],
                 hoverinfo='text',
                 name="Observation (KAF)")]

    layout_c = copy.deepcopy(layout)
    layout_c['title'] = (site_choice + " streamflow history ")
    layout_c['dragmode'] = 'select'
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['annotations'] = [annotation]
    layout_c['yaxis'] = yaxis
    layout_c['showlegend'] = True
    figure = dict(data=data, layout=layout_c)

    return figure


@app.callback(Output('err_evolve', 'figure'),
              [Input('year', 'value'),
               Input('site_choice', 'value')])
def makeGraph3(year, site_choice):
    if not site_choice:
        site_choice = 'MPHC2'
    dfs = df_dict[site_choice]
    df = dfs[year]
    df = df[75:295]  # January to August
    q = df['Observed Total'].dropna().tolist()[0]
    forecasts = np.array(df['ESP 50'])
    errors = abs(forecasts - q)
    errors = np.round(errors, 2)
    mean_err = round(np.nanmean(errors), 2)
    df['text'] = errors
    df['text'] = df['text'].astype(str) + " KAF"
    yaxis = dict(range=[0, 700])
    xaxis = dict(df['Date'])
    annotation = dict(
        text="<b>Mean Absolute Error: " + "{:,}".format(mean_err) + "</b>",
        x=df.Date.iloc[-15],
        y=650,
        font=dict(size=17),
        showarrow=False)

    data = [dict(type='line',
                 line=dict(color='red',
                           width=5),
                 x=df['Date'],
                 y=errors,
                 text=df['text'],
                 hoverinfo='text',
                 name="Error (p50 - q)")]

    layout_c = copy.deepcopy(layout)
    layout_c['title'] = (site_choice + " ESP 50 Absolute Errors " + year)
    layout_c['dragmode'] = 'select'
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['annotations'] = [annotation]
    layout_c['yaxis'] = yaxis
    layout_c['xaxis'] = xaxis
    layout_c['showlegend'] = True
    figure = dict(data=data, layout=layout_c)

    return figure


@app.callback(Output('uncrtnty_evolve', 'figure'),
              [Input('year', 'value'),
               Input('site_choice', 'value')])
def makeGraph4(year, site_choice):
    if not site_choice:
        site_choice = 'MPHC2'
    dfs = df_dict[site_choice]
    df = dfs[year]
    df = df[75:295]  # January to August
    uncertainties = np.array(df['ESP 10']) - np.array(df['ESP 90'])
    uncertainties = np.round(uncertainties, 2)
    df['text'] = uncertainties
    df['text'] = df['text'].astype(str) + " KAF"
    mean_uncert = round(np.nanmean(uncertainties), 2)
    yaxis = dict(range=[0, 700])
    annotation = dict(
        text="<b>Average Uncertainty: " + "{:,}".format(mean_uncert) + "</b>",
        x=df.Date.iloc[-15],
        y=650,
        font=dict(size=17),
        showarrow=False)

    data = [dict(type='line',
                 line=dict(color='#f4d942',
                           width=5),
                 x=df.Date,
                 y=uncertainties,
                 text=df['text'],
                 hoverinfo='text',
                 name="Uncertainty (p10 - p90)")]

    layout_c = copy.deepcopy(layout)
    layout_c['title'] = (site_choice + " ESP 90, 10 range " + year)
    layout_c['dragmode'] = 'select'
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['annotations'] = [annotation]
    layout_c['yaxis'] = yaxis
    layout_c['showlegend'] = True

    figure = dict(data=data, layout=layout_c)

    return figure


@app.callback(Output('err_evolve_all', 'figure'),
              [Input('site_choice', 'value')])
def makeGraph5(site_choice):
    if not site_choice:
        site_choice = 'MPHC2'
    dfs = df_dict[site_choice]
    dfs = [dfs[key] for key in dfs.keys()]
    for i in range(len(dfs)):
        dfs[i] = dfs[i][75:295]
        print(len(dfs[i]))  # January to August

    qs = [df['Observed Total'].dropna().tolist()[0] for df in dfs]
    forecasts = [np.array(df['ESP 50']) for df in dfs]
    errors = [forecasts[i] - qs[i] for i in range(len(qs))]
    errors = np.nanmean(errors, axis=0)
    # errors = abs(forecasts - q)
    errors = np.round(errors, 2)
    mean_err = round(np.nanmean(errors), 2)
    df = dfs[0][['Date', 'Average']]
    # df['Date'] = pd.to_datetime(df['Date'])
    # df['Date'] = df['Date'].map(lambda x: x.strftime('%m-%d'))
    df['day'] = df.index
    df['errors'] = errors
    df['text'] = df['errors'].astype(str) + " KAF"
    yaxis = dict(range=[0, 700])
    xaxis = dict(df['Date'])
    annotation = dict(
        text="<b>Mean Absolute Error: " + "{:,}".format(mean_err) + "</b>",
        x=330,
        y=650,
        font=dict(size=17),
        showarrow=False)

    data = [dict(type='line',
                 line=dict(color='red',
                           width=5),
                 x=df['day'],
                 y=df['errors'],
                 text=df['text'],
                 hoverinfo='text',
                 name="Error (p50 - q)")]

    layout_c = copy.deepcopy(layout)
    layout_c['title'] = (site_choice + " ESP 50 Absolute Errors - All Years")
    layout_c['dragmode'] = 'select'
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['annotations'] = [annotation]
    layout_c['yaxis'] = yaxis
    layout_c['xaxis'] = xaxis
    layout_c['showlegend'] = True
    figure = dict(data=data, layout=layout_c)

    return figure


@app.callback(Output('uncrtnty_evolve_all', 'figure'),
              [Input('year', 'value'),
               Input('site_choice', 'value')])
def makeGraph6(year, site_choice):
    if not site_choice:
        site_choice = 'MPHC2'
    dfs = df_dict[site_choice]
    dfs = [dfs[key] for key in dfs.keys()]
    for i in range(len(dfs)):
        dfs[i] = dfs[i][75:295]
        print(len(dfs[i]))  # January to August

    uncertainties = [np.array(df['ESP 10']) - np.array(df['ESP 90']) for
                     df in dfs]
    uncertainties = np.nanmean(uncertainties, axis=0)
    uncertainties = np.round(uncertainties, 2)
    df = dfs[0][['Date', 'Average']]
    df['uncertainties'] = uncertainties
    df['text'] = df['uncertainties'].astype(str) + " KAF"
    df['day'] = df.index
    mean_uncert = round(np.nanmean(uncertainties), 2)
    yaxis = dict(range=[0, 700])
    annotation = dict(
        text="<b>Average Uncertainty: " + "{:,}".format(mean_uncert) + "</b>",
        x=330,
        y=650,
        font=dict(size=17),
        showarrow=False)

    data = [dict(type='line',
                 line=dict(color='#f4d942',
                           width=5),
                 x=df.day,
                 y=df.uncertainties,
                 text=df['text'],
                 hoverinfo='text',
                 name="Uncertainty (p10 - p90)")]

    layout_c = copy.deepcopy(layout)
    layout_c['title'] = (site_choice + " ESP 90, 10 range - All Years")
    layout_c['dragmode'] = 'select'
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['annotations'] = [annotation]
    layout_c['yaxis'] = yaxis
    layout_c['showlegend'] = True

    figure = dict(data=data, layout=layout_c)

    return figure

# In[] Set up application callbacks for our simulation
@app.callback(Output('our_graph', 'figure'),
              [Input('year', 'value'),
               Input('site_choice', 'value'),
               Input('sd', 'value')])
def makeGraph7(year, site_choice, sd):
    if not site_choice:
        site_choice = 'MPHC2'
    dfs = df_dict[site_choice]
    df = dfs[year]
    df = df[75:295]  # January to August
    dates = df['Date']
    obs = df['Observed Accumulation'].dropna()
    final = obs.iloc[-1]
    average = list(df['Average'])[0]
    yaxis = dict(range=[0, 700])
    df['ratio50'] = df['ESP 50'].apply(
            lambda x: str(round(x/final*100, 2)) + "%")

# Simulation part:
    forecast = np.random.normal(final, sd, len(dates))

    df_f = pd.DataFrame({'Date': dates, 'Forecast': forecast})
    data = [dict(type='line',
                 line=dict(color='#cc872e',
                           width=4),
                 x=df_f.Date,
                 y=df_f.Forecast,
                 text=df['ratio50'],
                 hovermode='text',
                 name='Simulation'),
            dict(
              type='line',
              line=dict(color='blue',
                        width=4),
              x=df.Date,
              y=df['Observed Accumulation'],
              name='Observation')]

    layout_c = copy.deepcopy(layout)
    layout_c['dragmode'] = 'select'
    layout_c['yaxis'] = yaxis
    layout_c['title'] = 'Our "ESP" - ' + site_choice
    layout_c['font'] = dict(color='white'),
    layout_c['titlefont'] = dict(color='white',
                                 size='20',
                                 weight='bold')
    layout_c['legend'] = dict(font=dict(size=15),
                              orientation='v')
    layout_c['paper_bgcolor'] = '#013589'
    layout_c['showlegend'] = True
    figure = dict(data=data, layout=layout_c)

    return figure


# In[] Run application

@app.callback(Output('sd_output', 'children'),
              [Input('sd', 'value')])
def displaySD(sd):
    return str(sd)

@app.callback(Output('sd_output2', 'children'),
              [Input('sd2', 'value')])
def displaySD2(sd):
    return str(sd)

# In[]



# In[] 

if __name__ == '__main__':
    app.run_server()
