# -*- coding: utf-8 -*-
"""
Visualize the ensemble streamflow predictions

@author: Travis Williams
"""

from sys import platform
import copy
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import glob
import json
import numpy as np
import os
import pandas as pd
import random
import scipy

if platform == 'win32':
    os.chdir("..")
    from flask_cache import Cache
else:
    from flask_caching import Cache

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
        l=55,
        r=35,
        b=65,
        t=55,
        pad=4
    ),
    # hovermode="closest",
    plot_bgcolor="white",
    paper_bgcolor="#a6b727",
    legend=dict(font=dict(size=10), orientation='h'))

# In[] Get data
files = glob.glob(os.path.join('data', "*"))
esp_files = [f for f in files if "historical" not in f]
hist_files = [f for f in files if "historical" in f]
sites = np.unique([os.path.basename(f)[:-9] for f in esp_files])
hist_dict = {site: pd.read_csv(
                        os.path.join('data',
                                     site +
                                     "_historical.csv")) for site in sites}

# Dictionary of dataframes embedded according to place and year
df_dict = {}
for site in sites:
    site_files = [f for f in esp_files if site in f]
    site_dfs = {f[-8:-4]: pd.read_csv(f) for f in site_files}
    df_dict[site] = site_dfs

# Get available years
yrs = np.unique([f[-8:-4] for f in esp_files])

# For y-scales lets get the max historical values for each site, and scale up
y_scales = {site: np.nanmax(hist_dict[site]['vol']) * 1.25 for site in sites}

# DASHify the options
year_options = [{'label': y, 'value': y} for y in yrs]
site_options = [{'label': site, 'value': site} for site in sites]

# In[] Helpers


def simForecast(original):
    '''
    My silly standin forecast simulator
    Based on the characteristics of a known ensemble streamflow prediction
        timeseries, we would like to simulate a random forecast. Then,
        eventually, specify improvements to certain parameters that determine
        skill.
    '''
    # dates = np.array(original['Date'])
    # forecast = np.array(original['ESP 50'])

    # There are missing forecasts
    for i in range(len(original)):
        if np.isnan(original[i]):
            original[i] = np.nanmean(original[i-3: i+3])

    # The final forecast is the target streamflow (q)
    q = original[-1]

    # 1: Variation in forecast steps taken looks like the correct
    # progression of uncertainty
    steps = np.diff(original)

    # The progression of errors look like the progression of forecasts
    errors = np.array([f - q for f in original])

    # 2: So if we take a moving window of standard deviation of the steps...
    variations = [list(steps[i-3:i]) for i in range(3, len(steps))]
    sds = np.array([np.std(v) for v in variations])

    # 3: And use this to simulate improvements in errors
    simprovements = np.array([np.random.normal(0, abs(sd)) for sd in sds])

    # While the errors themselves are a random walk toward the final?

    # 4: We choose a starting number, random with a certain range around q
    pct = .5  # so the number would fall with +- 50% of ultimate q (parameter)
    low = 1 - pct
    high = 1 + pct
    start = random.uniform(q * low, q * high)

    # 5: We get our initial error
    e1 = start - q

    # 6: We iteratively add modeled improvements to the next error
    errs = [e1]
    for i in range(len(simprovements) - 1):
        print("TEST" + str(simprovements[i]))
        if not np.isnan(simprovements[i]):
            err = errs[i] + simprovements[i]
        else:
            err = errs[i]
        errs.append(err)

    # 7: Now we add our vector of errors to q
    fsim = q + np.array(errs)
    print("Success")
    return fsim


# In[] HTML Layout
app.layout = html.Div([

        html.H2("Ensemble Streamflow Predictions at the McPhee Reservoir",
                style={'text-align': 'center'}),

        html.Div(className="row",
                 children=[html.Div([dcc.Dropdown(id='site_choice',
                                                  options=site_options,
                                                  placeholder="McPhee " +
                                                              "Reservoir")],
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

        html.Div(className="six columns",
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
                                     html.P('The Hoped For "Skill" Meter'),
                                     html.P(id='sd_output'),
                                     dcc.Slider(id='sd',
                                                min=0,
                                                max=25,
                                                step=1,
                                                value=0,
                                                updatemode='drag',
                                                # vertical=True,
                                                marks={0: {'label': '0'},
                                                       25: {'label': '25'}})]),
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
    df = df[75:295]  # January to August
    obs = df['Observed Accumulation'].dropna()
    final = obs.iloc[-1]
    var = round(np.nanvar(df['ESP 50']), 2)
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    df['ratio10'] = df['ESP 10'].apply(lambda x:
                                       str(round(x/final*100, 2)) + "%")
    df['text10'] = df['ESP 10'].astype(str) + " KAF; " + df['ratio10']
    df['ratio50'] = df['ESP 50'].apply(lambda x:
                                       str(round(x/final*100, 2)) + "%")
    df['text50'] = df['ESP 50'].astype(str) + " KAF; " + df['ratio50']
    df['ratio90'] = df['ESP 90'].apply(lambda x:
                                       str(round(x/final*100, 2)) + "%")
    df['text90'] = df['ESP 90'].astype(str) + " KAF; " + df['ratio90']

    annotation = dict(
        text="<b>Forecast Variance: " + "{:,}".format(var) + "</b>",
        x=year + '-06-25',
        y=y_scales[site_choice] * .9,
        font=dict(size=17),
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
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    df['text'] = df['vol'].astype(str) + "KAF"
    annotation = dict(
        text="<b>Streamflow Variance: " + "{:,}".format(var) + "</b>",
        x=2010,
        y=y_scales[site_choice] * .9,
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
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    xaxis = dict(df['Date'])
    annotation = dict(
        text="<b>Mean Absolute Error: " + "{:,}".format(mean_err) + "</b>",
        x=df.Date.iloc[-15],
        y=y_scales[site_choice] * .9,
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
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    annotation = dict(
        text="<b>Average Uncertainty: " + "{:,}".format(mean_uncert) + "</b>",
        x=df.Date.iloc[-15],
        y=y_scales[site_choice] * .9,
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
    errors = [abs(forecasts[i] - qs[i]) for i in range(len(qs))]
    errors = np.nanmean(errors, axis=0)
    errors = np.round(errors, 2)
    mean_err = round(np.nanmean(errors), 2)
    df = dfs[0][['Date', 'Average']]  # Using a sample to start
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(lambda x: x.strftime('%j'))
    # dates = [str(d) for d in df['Date']]
    df['day'] = df.index
    df['errors'] = errors
    df['text'] = df['errors'].astype(str) + " KAF"
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    xaxis = dict(title="Time Index")
    annotation = dict(
        text="<b>Mean Absolute Error: " + "{:,}".format(mean_err) + "</b>",
        x=330,
        y=y_scales[site_choice] * .9,
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
    df['date'] = df['Date'].apply(lambda x: x[-5:])
    df['date'].iloc[0:17] = df['date'].iloc[0:17] + " year 1"
    df['uncertainties'] = uncertainties
    df['text'] = df['uncertainties'].astype(str) + " KAF"
    df['day'] = df.index
    mean_uncert = round(np.nanmean(uncertainties), 2)
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    xaxis = dict(title='Time Index')
    annotation = dict(
        text="<b>Average Uncertainty: " + "{:,}".format(mean_uncert) + "</b>",
        x=330,
        y=y_scales[site_choice] * .9,
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
    layout_c['xaxis'] = xaxis
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
    yaxis = dict(range=[0, y_scales[site_choice]],
                 title='Thousand Acre Feet')
    df['ratio50'] = df['ESP 50'].apply(
            lambda x: str(round(x/final*100, 2)) + "%")

# Simulation part:
    original = np.array(df['ESP 50'])
    forecast = simForecast(original)
    len_diff = len(original) - len(forecast)
    forecast = pd.DataFrame({"Date": dates[len_diff:],
                             "Forecast": list(forecast)})
    print(forecast.Forecast)
    data = [dict(type='line',
                 line=dict(color='#cc872e',
                           width=4),
                 x=forecast.Date,
                 y=forecast.Forecast,
                 text=forecast['Forecast'],
                 hovermode='text',
                 name='Simulation'),
            # dict(
            #   type='line',
            #   line=dict(color='blue',
            #             width=4),
            #   x=forecast.Date,
            #   y=forecast['Forecast'].iloc[0],
            #   name='Observation')
           ]

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
    layout_c['paper_bgcolor'] = '#667021'
    layout_c['showlegend'] = True
    figure = dict(data=data, layout=layout_c)

    return figure


# In[] Run application

@app.callback(Output('sd_output', 'children'),
              [Input('sd', 'value')])
def displaySD(sd):
    return str(sd)


# @app.callback(Output('sd_output2', 'children'),
#               [Input('sd2', 'value')])
# def displaySD2(sd):
#     return str(sd)


# In[]

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
