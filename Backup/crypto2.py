## -------------------------------------------------------  Libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Group
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, acf

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
## ------------------------------------------------------- Data Collection pt.1
data1 = yf.download(tickers=['BTC-USD'], 
                       period = '24h', 
                       interval = '15m',
                       group_by = 'ticker')


data2 = yf.download(tickers=['ETH-USD'], 
                      period = '24h', 
                       interval = '15m',
                       group_by = 'ticker')


data3 = yf.download(tickers=['USDT-USD'], 
                      period = '24h', 
                       interval = '15m',
                       group_by = 'ticker')


data4 = yf.download(tickers=['BNB-USD'], 
                      period = '24h',  
                       interval = '15m',
                       group_by = 'ticker')


data5 = yf.download(tickers=['USDC-USD'], 
                      period = '24h',   
                       interval = '15m',
                       group_by = 'ticker')

data6 = yf.download(tickers=['ADA-USD'], 
                      period = '24h',   
                       interval = '15m',
                       group_by = 'ticker')
data7 = yf.download(tickers=['HEX-USD'], 
                      period = '24h',   
                       interval = '15m',
                       group_by = 'ticker')

     
    
## -------------------------------------------------------  Label the cryptocurrencies    
data1['Crypto'] = 'BTC-USD'
data2['Crypto'] = 'ETH-USD'
data3['Crypto'] = 'USDT-USD'
data4['Crypto'] = 'BNB-USD'
data5['Crypto'] = 'USDC-USD'
data6['Crypto'] = 'ADA-USD'
data7['Crypto'] = 'HEX-USD'

## -------------------------------------------------------  Merge the datasets

df = pd.concat([data1,data2,data3,data4,data5, data6, data7])

# Reset Index
df = df.reset_index()

df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Date'] = df['Datetime'].dt.date
df['Month'] = df['Datetime'].dt.month

# --------------------------------------------------Functions -------------------------------------------------------

def autocorrelation_plots(series):
    
    
    fig = make_subplots(rows=2, cols=1)


    corr_array = pacf((series).dropna(), alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    
    corr_array_acf = acf((series).dropna(), alpha=0.05)
    lower_acf = corr_array_acf[1][:,0] - corr_array_acf[0]
    upper_acf = corr_array_acf[1][:,1] - corr_array_acf[0]


    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f'),
                   row=1,col=1)

    for x in range(len(corr_array[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=corr_array[0], 
                             mode='markers', 
                             marker_color='#1f77b4',
                             marker_size=12),
                  row=1,col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=upper_y, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)'),
                 row=1,col=1)
                  

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=lower_y, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)'),
                 row=1,col=1)

    
    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array_acf[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f'), 
                   row=2,col=1)

    for x in range(len(corr_array_acf[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=corr_array_acf[0], 
                             mode='markers', 
                             marker_color='#1f77b4',
                       marker_size=12),
                  row=2,col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=upper_acf, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)'),
                  row =2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])),
                             y=lower_acf, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)'),
                  row=2,col=1)

    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,20])
    fig.update_yaxes(zerolinecolor='#000000')
    
    #title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(template='none', 
                      height = 600, 
                      width = 1800,
                      margin=dict(l=60, r=60, t=20, b=50))
        
    return fig





    
    
### ---------------------------------------------- Dashboard ------------------------------------------------

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


### ----------------------------------------------  Stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.sever


### ----------------------------------------------  App Layout
app.layout = html.Div(children=[
    html.H1(children='Cryptocurrency Time Series Analysis', style={'font-family': "Verdana",
                                                 'text-align': 'center'}),
    html.H4('GARCH & ARCH models', style={'textAlign': 'center'}),
    
    # Horizontal Line before the dropdown
    html.Hr(),
    
## Tabs  ------------------------------------------------------------------------------------------------------
    dcc.Tabs(id="tabs-graph", value='tab-1-example-graph', children=[
        
        # Tab 1
        dcc.Tab(label='Trend', 
                value='tab-1',
                style=tab_style, 
                selected_style=tab_selected_style),
        
        
        # Tab 2
        dcc.Tab(label='Volatility', 
                value='tab-2', 
                style=tab_style, 
                selected_style=tab_selected_style),
        
        # Tab 3
        dcc.Tab(label='Autocorrelation', 
                value='tab-3', 
                style=tab_style, 
                selected_style=tab_selected_style),
        
        # Tab 4
        dcc.Tab(label='Models', 
                value='tab-4', 
                style=tab_style, 
                selected_style=tab_selected_style),
    ]),
    
    html.Div(id='tabs-content'),

])


### ---------------------------------------------- Callback 1 for the tabs
@app.callback(Output('tabs-content', 'children'),
              Input('tabs-graph', 'value'))



### ---------------------------------------------- Function to update when the tabs are selected
def render_content(tab):

    if tab == 'tab-1':
        return html.Div([
            html.H3('', style={'textAlign': 'center'}), # Title
    # Create Dcc Dropdownmenu
    html.Div([
        dcc.Dropdown(id='crypto-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Crypto'].unique()],
                     placeholder="Select Cryptocurrency",
                     style=dict(
                         width='40%',
                         verticalAlign="middle"),
                     value='BTC-USD')],),
    
### ---------------------------------------------- Graph
    dcc.Graph(id='price-graph'),])
    
    
    # Tab 2 
    elif tab == 'tab-2':
        return html.Div([
            html.H3('', style={'textAlign': 'center'}), # Title
    # Create Dcc Dropdownmenu
    html.Div([
        dcc.Dropdown(id='crypto-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Crypto'].unique()],
                     placeholder="Select Cryptocurrency",
                     style=dict(
                         width='40%',
                         verticalAlign="middle"),
                     value='BTC-USD')],),
            
            # Graph
            dcc.Graph(id='volatility-graph'
             ),

        ])
    
### ---------------------------------------------- Tab 3
    elif tab == 'tab-3':
        return html.Div([
            html.H3('', style={'textAlign': 'center'}), # Title
    # Create Dcc Dropdownmenu
    html.Div([
        dcc.Dropdown(id='crypto-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Crypto'].unique()],
                     placeholder="Select  Cryptocurrency",
                     style=dict(
                         width='40%',
                         verticalAlign="middle"),
                     value='BTC-USD')],),
    
            # Graph
            dcc.Graph(id='forecast-graph'
             ),
            
                    ])
            
            
    elif tab == 'tab-4':
        return html.Div([
            html.H4('', style={'textAlign': 'center'}), # Title
            
    # Create Dcc Dropdownmenu
    html.Div([
        dcc.Dropdown(id='crypto-dropdown',
                     options=[{'label': i, 'value': i}
                              for i in df['Crypto'].unique()],
                     placeholder="Select  Cryptocurrency",
                     style=dict(
                         width='40%',
                         verticalAlign="center",
                         justifyContent = 'center'),
                     value='BTC-USD')],),
            
            html.Hr(),
            html.Br(),
            
            # Slider
            html.Div([
                dcc.Slider(
                    id='3DHR Slider',
                    min=0,
                    max=10,
                    value=1,
                    marks={i: f"p {i}" for i in range(1,10, 1)},
                    step=1),
    html.Div(id="sliderVal")
]),
    
      
                        # Slider 2
            html.Div([
                dcc.Slider(
                    id='3DHR Slider2',
                    min=0,
                    max=10,
                    value=1,
                    marks={i: f"q {i}" for i in range(1,10, 1)},
                    step=1),
    html.Div(id="sliderVal2")
]),
            # Graph
            dcc.Graph(id='models-graph'
             ),

        ])



### ----------------------------------------------  Add a Callback to update the plots
@app.callback(
    Output(component_id='price-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value')
)



### ----------------------------------------------  Update Figure for Tab 1
def update_graph(selected_crypto):
    
    filtered_df = df[(df['Crypto'] == selected_crypto)][-20:]
                      
    
    roll_5 = filtered_df.Close.rolling(window=100, min_periods=1).mean()

    fig = make_subplots(rows=2, cols=1, 
                            horizontal_spacing=0.9, 
                            vertical_spacing=0.5)


### ----------------------------------------------  Add traces
    fig.add_trace(go.Candlestick(x=filtered_df['Datetime'],
                                 open=filtered_df['Open'],
                                 high=filtered_df['High'],
                                 low=filtered_df['Low']
                                 ,close=filtered_df['Close'],
                                 name = 'BTC-USD',
                                 increasing_line_color= '#1E90FF', 
                                 decreasing_line_color= '#F70D1A',
                                 legendgroup='1'),row=1, col=1)
    
    fig.add_trace(go.Scatter(x= filtered_df.Datetime, 
                             y = roll_5,
                             name = 'Moving Average',
                             marker_color='#000000',
                             legendgroup='1'),row=1, col=1)

    fig.add_trace(go.Bar(x=filtered_df.Datetime, 
                         y=filtered_df.Volume, 
                         name = "Volume",
                         legendgroup='2'
                        ),
                  row=2,col=1)

### ---------------------------------------------- Add widgets and slider range

    fig.update_layout(template = 'none', 
                      height = 600,
                      margin=dict(l=60, r=60, t=20, b=50),
                      hovermode="x",
                      legend_tracegroupgap = 400,
                      showlegend=True,
                      font_family="Verdana", # Set Font style
                      font_size=14) # Set Font size) # legend false 
    
    fig['layout']['yaxis1'].update(domain = [0.5, 1.0])
    

    return fig


### ---------------------------------------------- Add a Callback to update the plots in Tab 2
@app.callback(
    Output(component_id='volatility-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
)

def volatility(selected_crypto):
    
    filtered_df_2 = df[df['Crypto'] == selected_crypto]
    filtered_df_2 = filtered_df_2.set_index('Datetime')
    returns = 100 * filtered_df_2.Close.pct_change().dropna()
    returns = returns.reset_index()
    volatility_trend = px.line(returns, x = returns.Datetime, y = returns.Close)
    volatility_trend.update_layout(template = 'none', 
                                   height = 550,
                                   margin=dict(l=60, r=60, t=20, b=50),
                                   hovermode="x",
                                   showlegend=False)
    volatility_trend.update_yaxes(title_text='Returns')
    volatility_trend.update_xaxes(title_text='Hour')
    
    
    return volatility_trend


### ---------------------------------------------- Add a Callback to update the plots in Tab 3


@app.callback(
    Output(component_id='forecast-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
)

    
def forecast(selected_crypto):
    filtered_df = df[(df['Crypto'] == selected_crypto)]
    returns = 100 * filtered_df.Close.pct_change().dropna()
    
    return autocorrelation_plots(returns**2)


### ---------------------------------------------- Add a Callback to update the plots in Tab 4 

pmin = 1
pmax = 10

@app.callback(
    Output(component_id='models-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
    Input(component_id='3DHR Slider', component_property='value'),
    Input(component_id='3DHR Slider2', component_property='value'),
)

    
def forecast(selected_crypto, p, q):
    
    filtered_df_3 = df[df['Crypto'] == selected_crypto]
    filtered_df_3 = filtered_df_3.set_index('Datetime')
    returns = 100 * filtered_df_3.Close.pct_change().dropna()

    #df_usdt = df_usdt.resample('4h').mean().dropna(how='all')

    
    model = arch_model(returns,
                       p=p, 
                       q=q, 
                       rescale=False)

    model_fit = model.fit()

    rolling_predictions = []
    test_size = 50

    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, p=p, q=q, rescale=False)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
        
    rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-50:])
        
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = returns.index, 
                                 y=returns[-50:],
                                 mode='lines',
                                 name='Returns'))

    fig.add_trace(go.Scatter(x = returns.index, 
                                 y=rolling_predictions[-50:],
                                 mode='lines',
                                 name = 'Volatility Trend'))

    fig.update_layout(template='none',
                      height= 350,
                      width = 1700,
                      margin=dict(l=60, r=60, t=20, b=50),
                      legend=dict(
                          yanchor="top",
                          y=0.99,
                          xanchor="left",
                          x=0.01))

    
    return fig

 
    
if __name__ == '__main__':
    app.run_server(debug=False)
