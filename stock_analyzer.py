import yfinance as yf
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
pio.renderers.default = "browser"


''' Function Definitions for Analysis '''
def close_plot(df, adj = True, normalize = False):
       
    if adj:
        close = df.loc[:,"Adj Close"].copy()
        title = "Adjusted Closing Prices"
    else:
        close = df.loc[:,"Close"].copy()
        title = "Closing Prices"
        
    if normalize:
        normclose = close.div(close.iloc[0]) #Normalizes data
        fig = px.line(normclose, 
                      x = normclose.index,
                      y = normclose.columns,
                      title = "Normalized " + title,      
                      template = 'plotly_dark') # Plotting Normalized closing data
        fig.update_layout(
            legend = dict(title = None, font = dict(size = 16)),
            title={
            'y':0.9,
            'x':0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
            hovermode = "x unified",
            xaxis_title = "Date",
            yaxis_title = "Normalized " + title + " (USD)"
            )
        fig.show()
    else:
        fig = px.line(close, 
                      x = close.index,
                      y = close.columns,
                      title = title,
                      template = 'plotly_dark') # Plotting Normalized closing data
        fig.update_layout(
            legend = dict(title = None, font = dict(size = 16)),
            title={
            'y':0.9,
            'x':0.5,
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
            hovermode = "x unified",
            xaxis_title = "Date",
            yaxis_title = title +  " (USD)"
            )
        fig.show()
        
def returns_plot(df, adj = True):
       
    if adj:
        close = df.loc[:,"Adj Close"].copy()
    else:
        close = df.loc[:,"Close"].copy()

    ret = close.pct_change().dropna()
    cum_ret  = ((1 + ret).cumprod() -1) * 100
    fig = px.line(cum_ret, template = 'plotly_dark')
    fig.update_layout(
        legend = dict(title = None, font = dict(size = 16)),
        title={
            'y':0.95,
            'x':0.5,
            'text': "Daily Cumulative Returns",
            'font': {'size': 24},
            'xanchor': 'center',
            'yanchor': 'top'},
        hovermode = "x unified",
        xaxis_title = "Date",
        yaxis_title = "% Returns")
  
    fig.show()  
    
def risk_return(df, adj = True, crypto = False):
    
    if adj:
        close = df.loc[:,"Adj Close"].copy()
    else:
        close = df.loc[:,"Close"].copy()
    
    if crypto:
        trading_days = 365
    else:
        trading_days  = 252

        
    ret = close.pct_change().dropna().mul(100) #Returns of each stock in terms of percent change
    summary = ret.describe().T.loc[:,["mean","std"]]
    summary["mean"] = summary["mean"]*trading_days # Multiply by number of trading days
    summary["std"]  = summary["std"]*np.sqrt(trading_days) # Multiply by number of trading dayss
    summary.rename(columns = {'mean':'Return', 'std':'Risk'}, inplace = True)  
    fig = px.scatter(summary, 
                     x = 'Risk', 
                     y = 'Return', 
                     title = "Annual Risk / Return",
                     text = summary.index,
                     template = 'plotly_dark')
    fig.update_traces(marker={'size': 15},
                      textposition='top center',
                      hoverlabel=dict(font=dict(size=20) ))
    fig.update_layout(
        legend = dict(title = None),
        title={
        'y':0.9,
        'x':0.5,
        'font': {'size': 24},
        'xanchor': 'center',
        'yanchor': 'top',},
        xaxis = dict(title = dict(font = dict(size = 20))),
        yaxis = dict(title = dict(font = dict(size = 20)))
        )
    fig.show()

def ret_corr(df, adj = True, crypto = False):
    
    if adj:
        close = df.loc[:,"Adj Close"].copy()
    else:
        close = df.loc[:,"Close"].copy()
    
    if crypto:
        trading_days = 365
    else:
        trading_days  = 252
        
    ret = close.pct_change().dropna().mul(100) #Returns of each stock in terms of percent change
    summary = ret.describe().T.loc[:,["mean","std"]]
    summary["mean"] = summary["mean"]*trading_days # Multiply by number of trading days
    summary["std"]  = summary["std"]*np.sqrt(trading_days) # Multiply by number of trading days
    summary.rename(columns = {'mean':'Return', 'std':'Risk'}, inplace = True) 
    
    fig = px.imshow(ret.corr(), text_auto=True, color_continuous_scale='tempo', template = 'plotly_dark', title = "Returns Correlation")
    fig.update_layout(
        legend = dict(title = None),
        title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
        )
    fig.show()

''' Test ''' 

