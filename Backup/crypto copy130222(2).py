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
from arch.univariate import arch_model
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    
    data8 = yf.download(tickers=['BCH-USD'], 
                          period = '24h',   
                           interval = '15m',
                           group_by = 'ticker')
    
    data9 = yf.download(tickers=['XRP-USD'], 
                          period = '24h',   
                           interval = '15m',
                           group_by = 'ticker')

    data10 = yf.download(tickers=['BUSD-USD'], 
                          period = '24h',   
                           interval = '15m',
                           group_by = 'ticker')
    data11 = yf.download(tickers=['SOL-USD'], 
                          period = '24h',   
                           interval = '15m',
                           group_by = 'ticker')
    data12 = yf.download(tickers=['DOGE-USD'], 
                          period = '24h',   
                           interval = '15m',
                           group_by = 'ticker')
    data13 = yf.download(tickers=['LUNA1-USD'], 
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
    data8['Crypto'] = 'BCH-USD'
    data9['Crypto'] = 'XRP-USD'
    data10['Crypto'] = 'BUSD-USD'
    data11['Crypto'] = 'SOL-USD'
    data12['Crypto'] = 'DOGE-USD'
    data13['Crypto'] = 'LUNA1-USD'
    
    # Create Returns for 24 hours
    data1['Returns']= 100 * data1.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data2['Returns']= 100 * data2.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data3['Returns']= 100 * data3.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data4['Returns']= 100 * data4.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data5['Returns']= 100 * data5.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data6['Returns']= 100 * data6.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data7['Returns']= 100 * data7.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data8['Returns']= 100 * data8.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data9['Returns']= 100 * data9.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data10['Returns']= 100 * data10.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data11['Returns']= 100 * data11.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data12['Returns']= 100 * data12.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    data13['Returns']= 100 * data13.Close.pct_change().dropna().rolling(98, min_periods=1).sum().round(4)
    
    # Create Growth
    data1['Growth'] = data1['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data2['Growth'] = data2['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data3['Growth'] = data3['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data4['Growth'] = data4['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data5['Growth'] = data5['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data6['Growth'] = data6['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data7['Growth'] = data7['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data8['Growth'] = data8['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data9['Growth'] = data9['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data10['Growth'] = data10['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data11['Growth'] = data11['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data12['Growth'] = data12['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')
    data13['Growth'] = data13['Returns'].apply(lambda x: '↗️' if x > 0 else '↘️')

    # Merge the datasets

    df = pd.concat([data1,data2,data3,data4,data5, data6, data7, data8, data9, data10, data11, data12, data13])

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
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Partial Autocorrelation", "Autocorrelation"))

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
                   row=1,col=2)

    for x in range(len(corr_array_acf[0]))]


    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=corr_array_acf[0], 
                             mode='markers', 
                             marker_color='#090059',
                             marker_size=12,
                             name = 'ACF'),
                  row=1,col=2)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])), 
                             y=upper_acf, 
                             mode='lines', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Upper Bound'),
                  row =1, col=2)

    fig.add_trace(go.Scatter(x=np.arange(len(corr_array_acf[0])),
                             y=lower_acf, 
                             mode='lines',
                             fillcolor='rgba(32, 146, 230,0.3)',
                             fill='tonexty', 
                             line_color='rgba(255,255,255,0)',
                             name = 'Lower Bound'),
                  row=1,col=2)

    
    
    
    # Update Figures
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,20],showspikes=True)
    fig.update_yaxes(zerolinecolor='#000000',showspikes=True)
    
    #title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(template='none', 
                      height = 300,
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
    data8 = df[df['Crypto'] == 'BCH-USD'].set_index('Datetime')
    data9 = df[df['Crypto'] == 'XRP-USD'].set_index('Datetime')
    data10 = df[df['Crypto'] == 'BUSD-USD'].set_index('Datetime')
    data11 = df[df['Crypto'] == 'SOL-USD'].set_index('Datetime')
    data12 = df[df['Crypto'] == 'DOGE-USD'].set_index('Datetime')
    data13 = df[df['Crypto'] == 'LUNA1-USD'].set_index('Datetime')

    data1 = data1['Close'].resample('1d').mean().dropna(how='all').round(4)
    data2 = data2['Close'].resample('1d').mean().dropna(how='all').round(4)
    data3 = data3['Close'].resample('1d').mean().dropna(how='all').round(4)
    data4 = data4['Close'].resample('1d').mean().dropna(how='all').round(4)
    data5 = data5['Close'].resample('1d').mean().dropna(how='all').round(4)
    data6 = data6['Close'].resample('1d').mean().dropna(how='all').round(4)
    data7 = data7['Close'].resample('1d').mean().dropna(how='all').round(4)
    data8 = data8['Close'].resample('1d').mean().dropna(how='all').round(4)
    data9 = data9['Close'].resample('1d').mean().dropna(how='all').round(4)
    data10 = data10['Close'].resample('1d').mean().dropna(how='all').round(4)
    data11 = data11['Close'].resample('1d').mean().dropna(how='all').round(4)
    data12 = data12['Close'].resample('1d').mean().dropna(how='all').round(4)
    data13 = data13['Close'].resample('1d').mean().dropna(how='all').round(4)



    fig_09_trace_01 = go.Indicator(
            mode = "number+delta",
            value = data1[1],
            title = {"text": "BTC <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data1[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 1})


    fig_09_trace_02 = go.Indicator(
            mode = "number+delta",
            value = data2[1],
            title = {"text": "ETH <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data2[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 2})


    fig_09_trace_03 = go.Indicator(
            mode = "number+delta",
            value = data3[1],
            title = {"text": "USDT <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data3[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 3})

    fig_09_trace_04 = go.Indicator(
            mode = "number+delta",
            value = data4[1],
            title = {"text": "BNB<br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data4[0].round(2), 'relative': True},
                domain = {'row': 1, 'column': 4})

    fig_09_trace_05 = go.Indicator(
            mode = "number+delta",
            value = data5[1],
            title = {"text": "USDC <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data5[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 5})




    fig_09_trace_06 = go.Indicator(
            mode = "number+delta",
            value = data6[1],
            title = {"text": "ADA <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data6[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 6})



    fig_09_trace_07 = go.Indicator(
            mode = "number+delta",
            value = data7[1],
            title = {"text": "HEX <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data7[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 7})
    
    fig_09_trace_08 = go.Indicator(
            mode = "number+delta",
            value = data8[1],
            title = {"text": "BCH <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data8[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 8})
    
    fig_09_trace_09 = go.Indicator(
            mode = "number+delta",
            value = data9[1],
            title = {"text": "XRP <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data9[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 9})
    
    fig_09_trace_10 = go.Indicator(
            mode = "number+delta",
            value = data10[1],
            title = {"text": "BUSD <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data10[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 10})

    fig_09_trace_11 = go.Indicator(
            mode = "number+delta",
            value = data11[1],
            title = {"text": "SOL <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data11[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 11})

    fig_09_trace_12 = go.Indicator(
            mode = "number+delta",
            value = data12[1],
            title = {"text": "DODGE <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data12[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 12})
 
    fig_09_trace_13 = go.Indicator(
            mode = "number+delta",
            value = data13[1],
            title = {"text": "LUNA1 <br><span style='font-size:0.8em;color:gray'>"},
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': data13[0].round(2), 'relative': True},
            domain = {'row': 1, 'column': 13})    


    fig_09 = make_subplots(
        rows=1,
        cols=13,
        specs=[[{'type' : 'indicator'}, 
                {'type' : 'indicator'}, 
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'},  
                {'type' : 'indicator'},
                {'type' : 'indicator'},
                {'type' : 'indicator'}, 
               
               ]])


    fig_09.append_trace(fig_09_trace_01, row=1, col=1)
    fig_09.append_trace(fig_09_trace_02, row=1, col=2)
    fig_09.append_trace(fig_09_trace_03, row = 1, col=3)
    fig_09.append_trace(fig_09_trace_04, row = 1, col=4)
    fig_09.append_trace(fig_09_trace_05, row = 1, col=5)
    fig_09.append_trace(fig_09_trace_06, row = 1, col=6)
    fig_09.append_trace(fig_09_trace_07, row = 1, col=7)
    fig_09.append_trace(fig_09_trace_08, row = 1, col=8)
    fig_09.append_trace(fig_09_trace_09, row = 1, col=9)
    fig_09.append_trace(fig_09_trace_10, row = 1, col=10)
    fig_09.append_trace(fig_09_trace_11, row = 1, col=11)
    fig_09.append_trace(fig_09_trace_12, row = 1, col=12)
    fig_09.append_trace(fig_09_trace_13, row = 1, col=13)


    fig_09.update_layout(margin=dict(l=0, r=0, t=30, b=30),
                         height=150)
    
    
    return fig_09


def treemap_vol():
    df_tree = df[['Crypto', 'Volume']].groupby('Crypto', as_index=False).mean()\
    .sort_values('Volume', ascending=False).round(2)
    #df_tree = df_tree.query("Crypto!='BTC-USD'")
    treemap_vol = px.treemap(df_tree, path=['Crypto'], 
                     values='Volume', color='Crypto',
                     color_continuous_scale='Viridis')

    treemap_vol.update_layout(
        title = 'Daily Cumulative Volume',
        template='none',
        margin=dict(l=0, r=0, t=50, b=70),
        height = 550,
        width = 800,
        font_family="Verdana", # Set Font style
        font_size=18) # Set Font size) # legend false 
    
    
    
    return treemap_vol


def bar_returns():
    
    df_returns = df[['Crypto', 'Returns']].groupby('Crypto', as_index=False).mean()\
    .sort_values('Returns', ascending=False).round(2)
    

    df_returns["Color"] = np.where(df_returns["Returns"]<0, 'red', 'green')
    bar_returns = go.Figure()
    bar_returns.add_trace(
        go.Bar(name='Returns',
               x=df_returns['Crypto'],
               y=df_returns['Returns'],
               text=df_returns['Returns'],
               marker_color=df_returns['Color']))
    bar_returns.update_layout(
        margin=dict(l=60, r=80, t=40, b=70),
        title = 'Cumulative Daily Return (%)',
        height = 550,
        width = 850,
        font_family="Verdana", # Set Font style
        font_size=18, # Set Font size) # legend false 
        template='none',
        showlegend=False)
    
    
    
    return bar_returns

def bar_volume():
    df_volume = df[['Crypto', 'Volume']].groupby('Crypto', as_index=False).mean()\
    .sort_values('Volume', ascending=False).round(2)
    
    #df_volume = df_volume.query("Crypto!='BTC-USD'")
    bar_volume = px.bar(df_volume, 
                             x='Crypto', 
                             y='Volume',
                             color='Crypto',
                             text_auto='.2s',
                             color_continuous_scale='Viridis')
    bar_volume.update_layout(
        margin=dict(l=60, r=100, t=60, b=70),
        title = 'Daily Average Volume ($)',
        height = 550,
        width = 800,
        font_family="Verdana", # Set Font style
        font_size=18, # Set Font size) # legend false 
        template='none',
        showlegend=False)
    
    bar_volume.update_xaxes(title_text='')
    bar_volume.update_yaxes(title_text='')
    
    
    
    return bar_volume 


def market_actual():
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = df.Returns.sum() / df.Crypto.count(),
    domain = {'x': [0, 1], 'y': [0, 1]},
    number = {'suffix': "%"},
    delta = {'reference': 0},
    gauge = {
        'axis': {'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps' : [
                 {'range': [0, 20], 'color': "#157DEC"},
                 {'range': [-20, 0], 'color': "#C35817"}]}))
    
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=10),height=200)

    return fig

#def indicator_aic():
    
   #fig = go.Figure(go.Indicator(
        #mode = "number+gauge+delta",
        #gauge = {'shape': "bullet"},
       # value = 220,
        #domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
       #title = {'text': "Avg order size"}))

    #return fig

# Create a variable and call the function
df = get_data()

df['Date'] =  df.Datetime.dt.strftime('%d/%m/%y %H:%M')
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Adj Close','Volume']].round(4)

df_2 = df[['Date', 'Crypto', 'Adj Close', 'Returns', 'Growth',  'Volume']].sort_values('Date', ascending=False)
df_2[['Adj Close', 'Returns', 'Growth',  'Volume']] = df_2[['Adj Close', 'Returns', 'Growth',  'Volume']].round(4)
    

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
    
    dbc.Row( html.Marquee("Yahoo Finance Live Coin Watch "), style={'font-family': "Verdana", 
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
       # dcc.Tab(label='Volatility', 
                #value='tab-2', 
               #style=tab_style, 
                #selected_style=tab_selected_style),
        
        # Tab 3
        dcc.Tab(label='ARCH Models', 
                value='tab-3', 
                style=tab_style, 
                selected_style=tab_selected_style),
        
        # Tab 4
        #dcc.Tab(label='ARCH & GARCH Models', 
                #value='tab-4', 
                #style=tab_style, 
                #selected_style=tab_selected_style),
    ]),
    
    html.Div(id='tabs-content'),

])



@app.callback(Output('tabs-content', 'children'),
              Input('tabs-graph', 'value'))


def render_content(tab):
    
        # Tab 0
    if tab == 'tab-0':
        return html.Div([
            html.H3('Cryptocurrency Market Average Return', style={'textAlign': 'center',
                                                                   'font-family': 'verdana'}), # Title
            dbc.Row([
                html.Div(dcc.Graph(id = 'my_scatter_plot', figure=market_actual()))]),            
            
            
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "color": "#FEC700"}),
            html.Br(),
            html.Br(),
    # Create Dcc Dropdownmenu      
            dbc.Row([
                html.Div(dcc.Graph(id = 'my_scatter_plot', figure=indicators()))]),
            
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "color": "#FEC700"}),
            
            html.Br(),
            
            html.Div(children=[
                html.Div(
                    dash_table.DataTable(
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 16,
                    id = 'my_accounting_table',
                    data = df_2.to_dict('records'),
                    columns = [{'id': c, 'name':c} for c in df_2.columns],
                    style_table={ 'font-family': 'verdana', 
                                 'width': '800px',
                                 'height': '550px',
                                 'font-size': '18px'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{Returns} > 0',
                                'column_id': 'Returns'
                            },
                            'backgroundColor': '#157DEC',
                            'color': '#FFFFFF'},
                        {
                            'if': {
                                'filter_query': '{Returns} < 0',
                                'column_id': 'Returns'
                            },
                            'backgroundColor': '#C35817',
                            'color': '#FFFFFF'},                
                        {
                            'if': {
                                'filter_query': '{Returns} = 0',
                                'column_id': 'Returns'
                            },
                            'backgroundColor': '#FFF380',
                            'color': '#000000'}   
                        
                    
                    
                    
                    ]), 
                        
                    style={'display': 'inline-block',
                           'margin-left': '30px'}),
                html.Div(
                    dcc.Graph(
                        figure=bar_returns()), style={'display': 'inline-block',
                                                      'margin-left': '80px'}),  

                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),  
                html.Br(),
                html.Div(
                    dcc.Graph(
                        figure=treemap_vol()), style={'display': 'inline-block',
                                                      'margin-left': '30px'}),
       

                html.Div(
                    dcc.Graph(
                        figure=bar_volume()), style={'display': 'inline-block',
                                                     'margin-left': '100px'}),
            

            
            ], style={'display': 'inline-block'})])


    

    

    
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
    


                      
    elif tab == 'tab-3':
        return html.Div([
            html.H4('', style={'textAlign': 'center'}), # Title
            
    # Crypto dropdown
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
            
            # Break Line
            html.Br(),
            
            html.Div(dcc.Graph(id='adfuller'), style={'display': 'inline-block'}),

            # Volatility Graph
            html.Div(dcc.Graph(id='volatility-graph')),
            
            

            
            html.Br(),

            
            # Forecast Graph
            html.Div(dcc.Graph(id='forecast-graph'), style={'display': 'inline-block'}),

            
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "color": "#FEC700"}),
            
            # Break Lines
            html.Br(),
            html.Br(),
            
            html.Div(
                html.Div(
                    className='row', children=[
          
        # Dropdowns 1
            
                html.Div([
                    html.Div([
                    html.Label(['Models:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                        dcc.Dropdown(id='vol-dropdown',
                                     options=[{'label': 'GARCH', 'value': 'GARCH'},
                                              {'label': 'ARCH', 'value': 'ARCH'},
                                              {'label': 'EGARCH', 'value': 'EGARCH'},                            
                                              {'label': 'FIARCH', 'value': 'FIARCH'},
                                              {'label': 'HARCH', 'value': 'HARCH'}                            
                                 ],
                                     placeholder="Select  Model",
                                     style=dict(
                                     width='150px',
                                     verticalAlign="middle",
                                     justifyContent = 'center',
                                     fontFamliy = 'verdana'),
                                     value='GARCH')],
                        style=dict(display='inline-block')),

                    html.Div([
                    html.Label(['Mean:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                        dcc.Dropdown(id='mean-dropdown',
                                     options=[{'label': 'Constant', 'value': 'Constant'},
                                              {'label': 'Zero', 'value': 'Zero'},
                                              {'label': 'LS', 'value': 'LS'},                            
                                              {'label': 'AR', 'value': 'AR'},
                                              {'label': 'ARX', 'value': 'ARX'},
                                              {'label': 'HAR', 'value': 'HAR'},
                                              {'label': 'HARX', 'value': 'HARX'}  
                                 ],
                                     placeholder="Select Mean",
                                     style=dict(
                                     width='150px',
                                     verticalAlign="center",
                                     justifyContent = 'center',
                                     fontFamliy = 'Verdana'),
                                     value='Constant')],style=dict(display='inline-block')),

                    html.Div([
                    html.Label(['Distribution:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                        dcc.Dropdown(id='dist-dropdown',
                                     options=[{'label': 'Normal', 'value': 'normal'},
                                              {'label': 'Students', 'value': 't'},
                                              {'label': 'LS', 'value': 'LS'},                            
                                              {'label': 'Skewed Student’s t', 'value': 'skewt'},
                                              {'label': 'Generalized Error Distribution', 'value': 'generalized error'}
                                 ],
                                     placeholder="Select  Distribution",
                                     style=dict(
                                     width='150px',
                                     verticalAlign="middle",
                                     justifyContent = 'center',
                                     fontFamliy = 'Verdana'),
                                     value='normal', )],style=dict(display='inline-block')),],),


                # Sliders (p,o,q, power) 1
                html.Div([
                    html.Div([
                    html.Label(['Symmetric innovation:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                    daq.Slider(
                        id='3DHR Slider',
                        min=0,
                        max=20,
                        value=1,
                        #marks={i: f"p {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "p"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top':'25px'}),
                    html.Div([
                    html.Label(['Asymmetric innovation:'], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                    daq.Slider(
                        id='3DHR Slider2',
                        min=0,
                        max=20,
                        value=0,
                        #marks={i: f"d {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "o"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top': '50px'}),

                    html.Div([
                    html.Label(['Volatility:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),

                        daq.Slider(
                                id='3DHR Slider3',
                                min=0,
                                max=20,
                                value=0,
                                #marks={i: f"d {i}" for i in range(0,20, 1)},
                                handleLabel={"showCurrentValue": True,"label": "q"},
                                color = 'black',
                                step=1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top': '50px'}),

                    html.Div([
                    html.Label(['Power:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),

                    daq.Slider(
                        id='Power Slider',
                        min=1.0,
                        max=4.0,
                        value=2.0,
                        #marks={i: f"p {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "Power"},
                        color = 'black',
                        step=0.1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top': '50px'}),]),],
                    style=dict(display='flex')),style={'display': 'inline-block'}),
            
            
# --------------------
            
                        html.Div(
                html.Div(
                    className='row', children=[
          
        # Dropdowns 1
            
                html.Div([
                    html.Div([
                    html.Label(['Models:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                        dcc.Dropdown(id='vol-dropdown-2',
                                     options=[{'label': 'GARCH', 'value': 'GARCH'},
                                              {'label': 'ARCH', 'value': 'ARCH'},
                                              {'label': 'EGARCH', 'value': 'EGARCH'},                            
                                              {'label': 'FIARCH', 'value': 'FIARCH'},
                                              {'label': 'HARCH', 'value': 'HARCH'}                            
                                 ],
                                     placeholder="Select  Model",
                                     style=dict(
                                     width='150px',
                                     verticalAlign="middle",
                                     justifyContent = 'center',
                                     fontFamliy = 'verdana'),
                                     value='GARCH')],
                        style=dict(display='inline-block')),

                    html.Div([
                    html.Label(['Mean:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                        dcc.Dropdown(id='mean-dropdown-2',
                                     options=[{'label': 'Constant', 'value': 'Constant'},
                                              {'label': 'Zero', 'value': 'Zero'},
                                              {'label': 'LS', 'value': 'LS'},                            
                                              {'label': 'AR', 'value': 'AR'},
                                              {'label': 'ARX', 'value': 'ARX'},
                                              {'label': 'HAR', 'value': 'HAR'},
                                              {'label': 'HARX', 'value': 'HARX'}  
                                 ],
                                     placeholder="Select Mean",
                                     style=dict(
                                     width='150px',
                                     verticalAlign="center",
                                     justifyContent = 'center',
                                     fontFamliy = 'Verdana'),
                                     value='Constant')],style=dict(display='inline-block')),

                    html.Div([
                    html.Label(['Distribution:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                        dcc.Dropdown(id='dist-dropdown-2',
                                     options=[{'label': 'Normal', 'value': 'normal'},
                                              {'label': 'Students', 'value': 't'},
                                              {'label': 'LS', 'value': 'LS'},                            
                                              {'label': 'Skewed Student’s t', 'value': 'skewt'},
                                              {'label': 'Generalized Error Distribution', 'value': 'generalized error'}
                                 ],
                                     placeholder="Select  Distribution",
                                     style=dict(
                                     width='150px',
                                     verticalAlign="middle",
                                     justifyContent = 'center',
                                     fontFamliy = 'Verdana'),
                                     value='normal', )],style=dict(display='inline-block')),],),


                # Sliders (p,o,q, power) 1
                html.Div([
                    html.Div([
                    html.Label(['Symmetric innovation:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                    daq.Slider(
                        id='3DHR Slider-2',
                        min=0,
                        max=20,
                        value=1,
                        #marks={i: f"p {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "p"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top':'25px'}),
                    html.Div([
                    html.Label(['Asymmetric innovation:'], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),
                    daq.Slider(
                        id='3DHR Slider2-2',
                        min=0,
                        max=20,
                        value=0,
                        #marks={i: f"d {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "o"},
                        color = 'black',
                        step=1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top': '50px'}),

                    html.Div([
                    html.Label(['Volatility:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),

                        daq.Slider(
                                id='3DHR Slider3-2',
                                min=0,
                                max=20,
                                value=0,
                                #marks={i: f"d {i}" for i in range(0,20, 1)},
                                handleLabel={"showCurrentValue": True,"label": "q"},
                                color = 'black',
                                step=1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top': '50px'}),

                    html.Div([
                    html.Label(['Power:  '], style={'font-weight': 'bold', 
                                                    "text-align": "center", 
                                                    'font-family': 'verdana'}),

                    daq.Slider(
                        id='Power Slider-2',
                        min=1.0,
                        max=4.0,
                        value=2.0,
                        #marks={i: f"p {i}" for i in range(0,20, 1)},
                        handleLabel={"showCurrentValue": True,"label": "Power"},
                        color = 'black',
                        step=0.1,
                        size=100)],
                        style={'display':'flex',
                               'margin-left':'20px',
                               'margin-top': '50px'}),]),],
                    style=dict(display='flex')),style={'display': 'inline-block', 'margin-left':'165px'}),
            
        
            
     
            
            
            
            # Break & Horizontal Lines
            html.Br(),
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "color": "#FEC700"}),
            html.Br(),
              
            # Indicators
            html.Div(
                dcc.Graph(id="results-ind"),style={'display': 'inline-block'}),
                
            html.Div(
                dcc.Graph(id="results-ind-2"),style={'display': 'inline-block',
                                                     'margin-left':'220px'}),
            
            html.Br(),
            
            
            # Call the Models graph
            html.Div(
                dcc.Graph(id='models-graph'),style={'display': 'inline-block'}),
            
            # Call the Models graph-2
            html.Div(dcc.Graph(id='models-graph-2'),style={'display': 'inline-block',
                                                     'margin-left':'0px'}),
    
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
                            vertical_spacing=0.5,
                        subplot_titles=("Trend", "Volume"))

    fig.add_trace(go.Candlestick(x=filtered_df['Datetime'],
                                 open=filtered_df['Open'],
                                 high=filtered_df['High'],
                                 low=filtered_df['Low']
                                 ,close=filtered_df['Close'],
                                 name = 'BTC-USD',
                                 increasing_line_color= '#0000CD', 
                                 decreasing_line_color= '#9F000F',
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
                         marker_color='#34A56F',
                        ),
                  row=2,col=1).update_layout(height=400)


    fig.update_annotations(font=dict(family="Verdana}", size=24))
    fig.update_layout(template = 'none', 
                      height = 750,
                      margin=dict(l=60, r=60, t=30, b=50),
                      hovermode="x",
                      legend_tracegroupgap = 400,
                      showlegend=False,
                      font_family="Verdana", # Set Font style
                      font_size=18) # Set Font size) # legend false 
    
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    
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
    
    # Variables to plug in the Confidence Interval Formula
    x = returns.Close
    mean_pred = returns.Close.mean() # average 
    alpha = 1.960 # Confidence Interval chosen was at 95%. Thus, based on the statistical table 1.960
    sqrt_n = np.sqrt(len(returns.Close)) # Square roots of the sample size
    
    # Set Upper and Lower bounds for the Predicted Value
    returns['lower_pred'] = x - alpha * (mean_pred / sqrt_n)

    returns['upper_pred'] = x + alpha * (mean_pred / sqrt_n)
   
    # Create Figure
    fig = go.Figure()

    # Add traces observed values 
    fig.add_trace(go.Scatter(x=returns.Datetime, y=returns.Close,
                        mode='lines',
                        line=dict(color='#090059', width=1),
                        name='Test Set'))


    # Add the Confidence Interval for the Lower Bounds on the test test
    fig.add_trace(go.Scatter(x=returns.Datetime, y=returns["lower_pred"],
                        marker=dict(color="#444"),
                        line=dict(width=2),
                        mode='lines',
                        fillcolor='blue',
                        fill='tonexty',
                        name='Lower Bound'))

    # Add the Confidence Interval for the Upper Bounds on the test test
    fig.add_trace(go.Scatter(x=returns.Datetime, y=returns["upper_pred"],
                        marker=dict(color="#444"),
                        line=dict(width=2),
                        mode='lines',
                        fillcolor='blue',
                        fill='tonexty',
                        name='Upper Bound'))


    # Use update_layout in order to define few configuration such as figure height and width, title, etc
    fig.update_layout(
        height=350, # Figure height
        width=1750, # Figure width
        title={
            'text': '', # Subplot main title
            'y':0.99, # Set main title y-axis position
            'x':0.5, # Set main title x-axis position
            'xanchor': 'center', # xachor position
            'yanchor': 'top'}, # yachor position 
        showlegend=False,
        font_family="Verdana", # Set Font style
        font_size=18) # Set Font size) # legend false 

    # Update Styling
    fig.update_layout(hovermode="x", 
                      template = 'none',
                      margin=dict(l=60, r=60, t=10, b=50))


    # Add Spikes
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    
    
    return fig



@app.callback(
    Output(component_id='adfuller', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
)

def adfuller(selected_crypto):
    from statsmodels.tsa.stattools import adfuller
    filtered_df_2 = df[df['Crypto'] == selected_crypto]
    filtered_df_2 = filtered_df_2.set_index('Datetime')
    returns = 100 * filtered_df_2.Close.pct_change().dropna()
    
    returns = returns.reset_index()
    CLOSE = returns.Close.values
    result = adfuller(CLOSE)

    adf = result[0]

    pvalue = result[1]

    fig = go.Figure(go.Indicator(
    mode = "number",
   # gauge = {'shape': "bullet"},
    value = round(pvalue,5),
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "Augumented Dickey-Fuller Test - pvalue:"}))
    
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=100),
                      height=200)

    return fig




@app.callback(
    Output(component_id='results-ind', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
    Input(component_id='mean-dropdown', component_property='value'),
    Input(component_id='3DHR Slider', component_property='value'),
    Input(component_id='3DHR Slider2', component_property='value'),
    Input(component_id='3DHR Slider3', component_property='value'),
    Input(component_id='vol-dropdown', component_property='value'),
    Input(component_id='Power Slider', component_property='value'),
    Input(component_id='dist-dropdown', component_property='value')

)

def bic(selected_crypto, mean, p, o, q, vol, power, dist):
    
    filtered_df_3 = df[df['Crypto'] == selected_crypto]
    filtered_df_3 = filtered_df_3.set_index('Datetime')
    returns = 100 * filtered_df_3.Close.pct_change().dropna()


    model = arch_model(returns,
                       mean=mean,
                       p=p, 
                       o=o,
                       q=q, 
                       vol=vol,
                       power=power,
                       dist=dist,
                       rescale=False)

    model_fit = model.fit()

    bic = model_fit.bic
    aic = model_fit.aic
    
    fig = go.Figure(go.Indicator(
    mode = "number",
    #gauge = {'shape': "bullet"},
    value = bic,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "BIC"}))
    
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                      height=100)

    return fig



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
    Input(component_id='mean-dropdown', component_property='value'),
    Input(component_id='3DHR Slider', component_property='value'),
    Input(component_id='3DHR Slider2', component_property='value'),
    Input(component_id='3DHR Slider3', component_property='value'),
    Input(component_id='vol-dropdown', component_property='value'),
    Input(component_id='Power Slider', component_property='value'),
    Input(component_id='dist-dropdown', component_property='value')

)

    
def forecast(selected_crypto, mean, p, o, q, vol, power, dist):
    
    filtered_df_3 = df[df['Crypto'] == selected_crypto]
    filtered_df_3 = filtered_df_3.set_index('Datetime')
    returns = 100 * filtered_df_3.Close.pct_change().dropna()


    model = arch_model(returns,
                       mean=mean,
                       p=p, 
                       o=o,
                       q=q, 
                       vol=vol,
                       power=power,
                       dist=dist,
                       rescale=False)

    model_fit = model.fit()

    rolling_predictions = []
    test_size = 90

    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, mean=mean, p=p, o=o, q=q, vol=vol, power=power, dist=dist, rescale=False)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=4)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
        
    rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-90:])
    
    qqplot_data = qqplot(rolling_predictions, line='s').gca().lines
    
    

    # Create Subplots
    fig = make_subplots(rows=2, cols=2 )
    
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
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
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
    fig.update_layout(title={
            'text': '', # Subplot main title
            'y':0.98, # Set main title y-axis position
            'x':0.5, # Set main title x-axis position
            'xanchor': 'center', # xachor position
            'yanchor': 'top'}, # yachor position 
                      template = 'none', 
                      height = 550,
                      width = 900,
                      margin=dict(l=60, r=60, t=50, b=50),
                      hovermode="x",
                      showlegend=False,
                      font_family="Verdana", # Set Font style
                      font_size=18) # Set Font size) # legend false 
                        
    fig.update_annotations(font=dict(family="Verdana}", size=1))
    
    #fig['layout']['yaxis1'].update(domain = [0.5, 1.0])

    # Return figure
    return fig


@app.callback(
    Output(component_id='results-ind-2', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
    Input(component_id='mean-dropdown', component_property='value'),
    Input(component_id='3DHR Slider-2', component_property='value'),
    Input(component_id='3DHR Slider2-2', component_property='value'),
    Input(component_id='3DHR Slider3-2', component_property='value'),
    Input(component_id='vol-dropdown-2', component_property='value'),
    Input(component_id='Power Slider-2', component_property='value'),
    Input(component_id='dist-dropdown-2', component_property='value')

)

def bic_2(selected_crypto, mean, p, o, q, vol, power, dist):
    
    filtered_df_3 = df[df['Crypto'] == selected_crypto]
    filtered_df_3 = filtered_df_3.set_index('Datetime')
    returns = 100 * filtered_df_3.Close.pct_change().dropna()


    model = arch_model(returns,
                       mean=mean,
                       p=p, 
                       o=o,
                       q=q, 
                       vol=vol,
                       power=power,
                       dist=dist,
                       rescale=False)

    model_fit = model.fit()

    bic = model_fit.bic
    aic = model_fit.aic
    
    fig = go.Figure(go.Indicator(
    mode = "number",
   # gauge = {'shape': "bullet"},
    value = bic,
    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
    title = {'text': "BIC"}))
    
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                      height=100)

    return fig

# -------------- 

@app.callback(
    Output(component_id='models-graph-2', component_property='figure'),
    Input(component_id='crypto-dropdown', component_property='value'),
    Input(component_id='mean-dropdown-2', component_property='value'),
    Input(component_id='3DHR Slider-2', component_property='value'),
    Input(component_id='3DHR Slider2-2', component_property='value'),
    Input(component_id='3DHR Slider3-2', component_property='value'),
    Input(component_id='vol-dropdown-2', component_property='value'),
    Input(component_id='Power Slider-2', component_property='value'),
    Input(component_id='dist-dropdown-2', component_property='value')

)

    
def forecast(selected_crypto, mean, p, o, q, vol, power, dist):
    
    filtered_df_3 = df[df['Crypto'] == selected_crypto]
    filtered_df_3 = filtered_df_3.set_index('Datetime')
    returns = 100 * filtered_df_3.Close.pct_change().dropna()


    model = arch_model(returns,
                       mean=mean,
                       p=p, 
                       o=o,
                       q=q, 
                       vol=vol,
                       power=power,
                       dist=dist,
                       rescale=False)

    model_fit = model.fit()

    rolling_predictions = []
    test_size = 90

    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, mean=mean, p=p, o=o, q=q, vol=vol, power=power, dist=dist, rescale=False)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=4)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
        
    rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-90:])
    
    qqplot_data = qqplot(rolling_predictions, line='s').gca().lines
    
    

    # Create Subplots
    fig = make_subplots(rows=2, cols=2 )
    
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
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
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
    fig.update_layout(title={
            'text': '', # Subplot main title
            'y':0.98, # Set main title y-axis position
            'x':0.5, # Set main title x-axis position
            'xanchor': 'center', # xachor position
            'yanchor': 'top'}, # yachor position 
                      template = 'none', 
                      height = 550,
                      width = 870,
                      margin=dict(l=60, r=60, t=50, b=50),
                      hovermode="x",
                      showlegend=False,
                      font_family="Verdana", # Set Font style
                      font_size=18) # Set Font size) # legend false 
                        
    fig.update_annotations(font=dict(family="Verdana}", size=1))
    
    #fig['layout']['yaxis1'].update(domain = [0.5, 1.0])

    # Return figure
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
