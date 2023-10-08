import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash import Dash, dash_table
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Group
import dash_bootstrap_components as dbc
from arch import arch_model
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, acf

# Create Dash App
app = dash.Dash(__name__)

# Variable to store the app server 
server = app.server

def get_data():

    # Data Collection pt.1
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

         
        
    # Label the cryptocurrencies    
    data1['Crypto'] = 'BTC-USD'
    data2['Crypto'] = 'ETH-USD'
    data3['Crypto'] = 'USDT-USD'
    data4['Crypto'] = 'BNB-USD'
    data5['Crypto'] = 'USDC-USD'
    data6['Crypto'] = 'ADA-USD'
    data7['Crypto'] = 'HEX-USD'
    

    # Merge the datasets

    df = pd.concat([data1,data2,data3,data4,data5, data6, data7])

    # Reset Index
    df = df.reset_index()

    df['Hour'] = df['Datetime'].dt.hour
    df['Day'] = df['Datetime'].dt.day
    df['Date'] = df['Datetime'].dt.date
    df['Month'] = df['Datetime'].dt.month

    return df


# Function to create the autocorrelation plots
def autocorrelation_plots(series):
    
    # Create subplots figure
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Partial Autocorrelation", "Autocorrelation"))

    # Partial Autocorrelation Plot (PACF)
    corr_array = pacf((series).dropna(), alpha=0.05) 
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
    
    # Autocorrelation Plot (ACF)
    corr_array_acf = acf((series).dropna(), alpha=0.05) 
    lower_acf = corr_array_acf[1][:,0] - corr_array_acf[0]
    upper_acf = corr_array_acf[1][:,1] - corr_array_acf[0]

    
    # Partial Autocorrelation Plot
    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f',
                              name = 'PACF'),
                   row=1,col=1)


    for x in range(len(corr_array[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=corr_array[0], 
                             mode='markers', 
                             marker_color='#090059',
                             marker_size=12,
                             name = 'PACF'),
                  row=1,col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=upper_y, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Upper Bound'),
                 row=1,col=1)
                  

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), 
                             y=lower_y, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Lower Bound'),
                 row=1,col=1)

    
    

    
    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array_acf[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f',
                              name = 'ACF'), 
                   row=2,col=1)

    for x in range(len(corr_array_acf[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=corr_array_acf[0], 
                             mode='markers', 
                             marker_color='#090059',
                             marker_size=12,
                             name = 'ACF'),
                  row=2,col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=upper_acf, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Upper Bound'),
                  row =2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])),
                             y=lower_acf, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Lower Bound'),
                  row=2,col=1)

    
    
    
    # Update Figures
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,20])
    fig.update_yaxes(zerolinecolor='#000000')
    
    #title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(template='none', 
                      height = 600, 
                      width = 1750,
                      margin=dict(l=60, r=60, t=50, b=50))
    fig.update_annotations(font=dict(family="Verdana}", size=24))
        
    # Return the subplots
    return fig

def indicators():
    
    data1 = df[df['Crypto'] == 'BTC-USD'].set_index('Datetime')
    data2 = df[df['Crypto'] == 'ETH-USD'].set_index('Datetime')
    data3= df[df['Crypto'] == 'USDT-USD'].set_index('Datetime')
    data4= df[df['Crypto'] =='BNB-USD'].set_index('Datetime')
    data5 = df[df['Crypto'] =='USDC-USD'].set_index('Datetime')
    data6= df[df['Crypto'] == 'ADA-USD'].set_index('Datetime')
    data7 = df[df['Crypto'] == 'HEX-USD'].set_index('Datetime')

    data1 = data1['Close'].resample('1d').mean().dropna(how='all')
    data2 = data2['Close'].resample('1d').mean().dropna(how='all')
    data3 = data3['Close'].resample('1d').mean().dropna(how='all')
    data4 = data4['Close'].resample('1d').mean().dropna(how='all')
    data5 = data5['Close'].resample('1d').mean().dropna(how='all')
    data6 = data6['Close'].resample('1d').mean().dropna(how='all')
    data7 = data7['Close'].resample('1d').mean().dropna(how='all')



    fig_09_trace_01 = go.Indicator(
            mode = "number+delta",
            value = data1[1],
            title = {"text": "BTC <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data1[0].round(2), 'relative': True, 'position' : "top"},
            domain = {'row': 1, 'column': 1})


    fig_09_trace_02 = go.Indicator(
            mode = "number+delta",
            value = data2[1],
            title = {"text": "ETH <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data2[0].round(2), 'relative': True, 'position' : "top"},
            domain = {'row': 1, 'column': 2})


    fig_09_trace_03 = go.Indicator(
            mode = "number+delta",
            value = data3[1],
            title = {"text": "USDT <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data3[0].round(2), 'relative': True, 'position' : "top"},
            domain = {'row': 1, 'column': 3})

    fig_09_trace_04 = go.Indicator(
            mode = "number+delta",
            value = data4[1],
            title = {"text": "BNB Price<br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data4[0].round(2), 'relative': True, 'position' : "top"},
                domain = {'row': 1, 'column': 4})

    fig_09_trace_05 = go.Indicator(
            mode = "number+delta",
            value = data5[1],
            title = {"text": "USDC <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data5[0].round(2), 'relative': True, 'position' : "top"},
            domain = {'row': 1, 'column': 5})




    fig_09_trace_06 = go.Indicator(
            mode = "number+delta",
            value = data6[1],
            title = {"text": "ADA <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data6[0].round(2), 'relative': True, 'position' : "top"},
            domain = {'row': 1, 'column': 6})



    fig_09_trace_07 = go.Indicator(
            mode = "number+delta",
            value = data7[1],
            title = {"text": "HEX <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data7[0].round(2), 'relative': True, 'position' : "top"},
            domain = {'row': 1, 'column': 7})


    fig_09 = make_subplots(
        rows=1,
        cols=7,
        specs=[[{'type' : 'indicator'}, 
                {'type' : 'indicator'}, 
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'}]])


    fig_09.append_trace(fig_09_trace_01, row=1, col=1)
    fig_09.append_trace(fig_09_trace_02, row=1, col=2)
    fig_09.append_trace(fig_09_trace_03, row = 1, col=3)
    fig_09.append_trace(fig_09_trace_04, row = 1, col=4)
    fig_09.append_trace(fig_09_trace_05, row = 1, col=5)
    fig_09.append_trace(fig_09_trace_06, row = 1, col=6)
    fig_09.append_trace(fig_09_trace_07, row = 1, col=7)


    fig_09.update_layout(margin=dict(l=60, r=10, t=10, b=10))
    
    return fig_09





# Create a variable and call the function
df = get_data()

df['Date'] =  df.Datetime.dt.strftime('%d/%m/%y %H:%M')
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].round(2)

df_2 = df[['Date', 'Crypto' ,'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    



# CSS Style
tabs_styles = {
    'height': '44px',
    'font-family': 'Verdana'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'font-family': 'Verdana'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'font-family': 'Verdana'
}





app.layout = html.Div(children=[
    html.H2(children='Cryptocurrency Time Series Analysis', style={'font-family': "Verdana", 
                                                                   'text-align': 'center',
                                                                   'color': '#090059'}),
    
    dbc.Row( html.Marquee("Live Coin Watch "), style={'font-family': "Verdana", 
                                                  'text-align': 'center', 
                                                  'color': '#090059'}),
    
    # Horizontal Line before the dropdown
    html.Hr(),
 
    
    dcc.Tabs(id="tabs-graph", value='tab-0', children=[
        
        # Tab 0
        dcc.Tab(label='Summary', 
                value='tab-0',
                style=tab_style, 
                selected_style=tab_selected_style),
        
        
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



@app.callback(Output('tabs-content', 'children'),
              Input('tabs-graph', 'value'))


def render_content(tab):
    
        # Tab 0
    if tab == 'tab-0':
        return html.Div([
            html.H3('', style={'textAlign': 'center'}), # Title
    # Create Dcc Dropdownmenu
    html.Div([dcc.Graph(id='price-graph' , 
                        figure= indicators())]),        
     
    html.Div([dash_table.DataTable(
    data=df.to_dict('records'),
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    columns=[
        {"name": i, "id": i} for i in df_2.columns],
        virtualization=True,
        fixed_rows={'headers': True},
        style_cell={'border': '1px solid grey' , 
                    'minWidth': 95, 'width': 95, 
                    'maxWidth': 95,
                    'fontSize':14, 
                    'font-family':'verdana'},
        style_table={ 'border': '1px solid black' , 
                     'height': 300, 
                     'font-family': 'verdana'}  # default is 500
    
    )])
    
    
        ])
    

    
    # Tab 1
    elif tab == 'tab-1':
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
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Verdana'),
                     value='BTC-USD'),
        
                html.Div(id="Dropdown_menu"),],
                style={'width': '50%',
                       'padding-left':'90%', 
                       'padding-right':'90%',
                       'display':'flex',
                       'margin': 'auto',
                       'font-family': 'verdana'}),
            
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
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Verdana'),
                     value='BTC-USD'),
        
                html.Div(id="Dropdown_menu"),],
                style={'width': '50%',
                       'padding-left':'90%', 
                       'padding-right':'90%',
                       'display':'flex',
                       'margin': 'auto',
                       'font-family': 'verdana'}),
            
            # Graph
            dcc.Graph(id='volatility-graph'
             ),

        ])
    

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
                         verticalAlign="center",
                         justifyContent = 'center',
                         fontFamliy = 'Verdana'),
                     value='BTC-USD'),
        
        
                html.Div(id="Dropdown_menu"),],
                style={'width': '50%',
                       'padding-left':'90%', 
                       'padding-right':'90%',
                       'display':'flex',
                       'margin': 'auto',
                       'font-family': 'verdana'}),
        
    
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
                         justifyContent = 'center',
                         fontFamliy = 'Verdana'),
                     value='BTC-USD'),

                html.Div(id="Dropdown_menu"),],
                style={'width': '50%',
                       'padding-left':'90%', 
                       'padding-right':'90%',
                       'display':'flex',
                       'margin': 'auto',
                       'font-family': 'verdana'}),
            
            # Slider 1
            html.Div([
                daq.Slider(
                    id='3DHR Slider',
                    min=0,
                    max=20,
                    value=1,
                    #marks={i: f"p {i}" for i in range(0,20, 1)},
                    handleLabel={"showCurrentValue": True,"label": "p"},
                    color = 'black',
                    step=1,
                    size=100),
                
                #daq.Slider(
                    #id='3DHR Slider2',
                    #min=0,
                    #max=20,
                    #value=1,
                   # marks={i: f"d {i}" for i in range(0,20, 1)},
                    #handleLabel={"showCurrentValue": True,"label": "d"},
                    #color = 'black',
                    #step=1,
                    #size=100),
                
                daq.Slider(
                    id='3DHR Slider3',
                    min=0,
                    max=20,
                    value=1,
                    marks={i: f"d {i}" for i in range(0,20, 1)},
                    handleLabel={"showCurrentValue": True,"label": "q"},
                    color = 'black',
                    step=1,
                    size=100),
                

    html.Div(id="sliderVal"),],
                style={'width': '50%',
                       'padding-left':'45%', 
                       'padding-right':'45%',
                       'display':'flex',
                       'margin': 'auto',
                       'font-family': 'verdana'}),
    
            
            
            
            # Graph
            dcc.Graph(id='models-graph'),

        ])
    


@app.callback(
    Output(component_id='price-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value')
)



def update_graph(selected_crypto):
    
    filtered_df = df[(df['Crypto'] == selected_crypto)][-20:]
                      
    
    roll_5 = filtered_df.Close.rolling(window=100, min_periods=1).mean()

    fig = make_subplots(rows=2, cols=1, 
                            horizontal_spacing=0.9, 
                            vertical_spacing=0.6,
                        subplot_titles=("Trend", "Volume"))

    fig.add_trace(go.Candlestick(x=filtered_df['Datetime'],
                                 open=filtered_df['Open'],
                                 high=filtered_df['High'],
                                 low=filtered_df['Low']
                                 ,close=filtered_df['Close'],
                                 name = 'BTC-USD',
                                 increasing_line_color= '#090059', 
                                 decreasing_line_color= 'red',
                                 legendgroup='1'),row=1, col=1)
    
    fig.add_trace(go.Scatter(x= filtered_df.Datetime, 
                             y = roll_5,
                             name = 'Moving Average',
                             marker_color='#000000',
                             legendgroup='1'),row=1, col=1).update_layout(height=300)


    fig.add_trace(go.Bar(x=filtered_df.Datetime, 
                         y=filtered_df.Volume, 
                         name = "Volume",
                         legendgroup='2',
                         marker_color='#357EC7',
                        ),
                  row=2,col=1).update_layout(height=400)


    fig.update_annotations(font=dict(family="Verdana}", size=24))
    fig.update_layout(template = 'none', 
                      height = 650,
                      margin=dict(l=60, r=60, t=30, b=50),
                      hovermode="x",
                      legend_tracegroupgap = 400,
                      showlegend=False,
                      font_family="Verdana", # Set Font style
                      font_size=14) # Set Font size) # legend false 
    
    fig['layout']['yaxis1'].update(domain = [0.6, 1])

    return fig


@app.callback(
    Output(component_id='volatility-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
)

def volatility(selected_crypto):
    
    filtered_df_2 = df[df['Crypto'] == selected_crypto]
    filtered_df_2 = filtered_df_2.set_index('Datetime')
    returns = 100 * filtered_df_2.Close.pct_change().dropna()
    returns = returns.reset_index()
    volatility_trend = px.line(returns, x = returns.Datetime, y = returns.Close, markers=True, color_discrete_sequence=["black"])
    volatility_trend.update_layout(template = 'none', 
                                   height = 600,
                                   margin=dict(l=60, r=60, t=20, b=50),
                                   hovermode="x",
                                   showlegend=False)
    volatility_trend.update_yaxes(title_text='Returns')
    volatility_trend.update_xaxes(title_text='Hour')
    
    
    return volatility_trend



@app.callback(
    Output(component_id='forecast-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
)

    
def acf_pacf(selected_crypto):
    filtered_df = df[(df['Crypto'] == selected_crypto)]
    returns = 100 * filtered_df.Close.pct_change().dropna()
    
    return autocorrelation_plots(returns**2)


@app.callback(
    Output(component_id='models-graph', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
    Input(component_id='3DHR Slider', component_property='value'),
    #Input(component_id='3DHR Slider2', component_property='value'),
    Input(component_id='3DHR Slider3', component_property='value'),
)

    
def forecast(selected_crypto, p, q):
    
    filtered_df_3 = df[df['Crypto'] == selected_crypto]
    filtered_df_3 = filtered_df_3.set_index('Datetime')
    returns = 100 * filtered_df_3.Close.pct_change().dropna()


    model = arch_model(returns,
                       p=p, 
                       q=q, 
                       rescale=False)

    model_fit = model.fit()

    rolling_predictions = []
    test_size = 90

    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, p=p, q=q, rescale=False)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=4)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
        
    rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-90:])
    
    qqplot_data = qqplot(rolling_predictions, line='s').gca().lines


    # Create Subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Returns vs Rolling Predictions", 
                                        "Histogram",
                                        "Q-Q Plot",
                                        "Autocorrelation Plot"))
    
    # Returns & Rolling Predictions
    fig.add_trace(go.Scatter(x = returns.index, 
                                 y=returns[-97:],
                                 mode='lines',
                                 name='Returns'), 
                  row=1, col=1)

    fig.add_trace(go.Scatter(x = returns.index, 
                                 y=rolling_predictions[-97:],
                                 mode='lines',
                                 name = 'Rolling Predictions',
                                 line=dict(color='#090059')), 
                  row=1, col=1)
    
    # Distplot
    hist_data = [rolling_predictions]
    group_labels = ['Rolling Predictions'] # name of the dataset

    distplfig = ff.create_distplot(hist_data, 
                                   group_labels, 
                                   bin_size=.01, 
                                   show_rug=False)
    # For Loop to plot Distplot
    for k in range(len(distplfig.data)):
        fig.add_trace(distplfig.data[k],
        row=2, col=1)
        
    
    # Q-Q plot
    fig.add_trace({
    'type': 'scatter',
    'x': qqplot_data[0].get_xdata(),
    'y': qqplot_data[0].get_ydata(),
    'mode': 'markers',
    'marker': {
        'color': '#090059'
    },
    'name': 'Q-Q Line'
    
    }, row=1, col=2)

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': 'red'
        },
        'name': 'Probability line'

    }, row=1, col=2)
    
    
    # ACF Plot
    
    # Autocorrelation Plot (ACF)
    corr_array_acf = acf((rolling_predictions).dropna(), alpha=0.05) 
    lower_acf = corr_array_acf[1][:,0] - corr_array_acf[0]
    upper_acf = corr_array_acf[1][:,1] - corr_array_acf[0]

    
    [fig.add_trace(go.Scatter(x=(x,x), 
                              y=(0,corr_array_acf[0][x]), 
                              mode='lines',
                              line_color='#3f3f3f',
                              name = 'ACF'), 
                   row=2,col=2)

    for x in range(len(corr_array_acf[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=corr_array_acf[0], 
                             mode='markers', 
                             marker_color='#090059',
                             marker_size=12,
                             name='ACF'),
                  row=2,col=2)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=upper_acf, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Upper bound'),
                  row =2, col=2)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])),
                             y=lower_acf, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)',
                             name='Lower Bound'),
                  row=2,col=2)

    # Figure Update Layout
    fig.update_layout(template = 'none', 
                      height = 650,
                      width = 1800,
                      margin=dict(l=60, r=60, t=50, b=50),
                      hovermode="x",
                      showlegend=False,
                      font_family="Verdana", # Set Font style
                      font_size=14) # Set Font size) # legend false 
                        
    fig.update_annotations(font=dict(family="Verdana}", size=1))
    
    #fig['layout']['yaxis1'].update(domain = [0.5, 1.0])

    # Return figure
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
