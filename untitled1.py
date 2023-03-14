
''' Portfolio Optimization '''

# import pandas as pd
import numpy as np
# from pypfopt import risk_models, plotting, expected_returns, EfficientFrontier as EF
import yfinance as yf
# from stock_analyzer import returns_plot
from datetime import datetime
# import yfinance as yf
import plotly.express as px
import plotly.io as pio
# import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import random

pio.renderers.default = "browser"

ticker ='SPY'
start_date = '2017-04-01' 
# stock_data = yf.download(tickers, start = start_date)



class Portfolio:
    def __init__(self, assets, start_date, weights = None, end_date = datetime.today().strftime('%Y-%m-%d')):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.generate_random_weights()
    def generate_random_weights(self):
    # Generate random weights that add up to 1 and round to 1 digit
        weights = [round(random.random(), 1) for _ in range(len(self.assets))]
        total_weight = sum(weights)
        return np.array([round(weight / total_weight, 1) for weight in weights])
            
      

    # Data gather using yfinance
    def get_data(self):
        df = yf.download(self.assets, start = self.start_date, end = self.end_date)
        return df
    
    # Plot the closing price data 
    def close_plot(self, adj = False, normalize = False, template = 'plotly_dark'):
        df = self.get_data()   
        if adj:
            close = df.loc[:,"Adj Close"].copy()
            title = "Adjusted Closing Prices"
        else:
            close = df.loc[:,"Close"].copy()
            title = "Closing Prices"
            
        if normalize:
            normclose = close.div(close.iloc[0]) #Normalizes data
            
            if isinstance(normclose, pd.Series): # This checks if the 
                normclose.name = self.assets[0]
                
            normclose = normclose.to_frame()
            fig = px.line(normclose, 
                          x = normclose.index,
                          y = normclose.columns,
                          title = "Normalized " + title,      
                          template = template) # Plotting Normalized closing data
            fig.update_traces(hovertemplate='%{y}')
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
            
            if isinstance(close, pd.Series): # This checks if the 
                close.name = self.assets[0]
            
            fig = px.line(close, 
                          x = close.index,
                          y = close.columns,
                          title = title,
                          template = template,
                          fill='tozeroy') # Plotting Normalized closing data
            fig.update_traces(hovertemplate='%{y}')
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
        
    
    def simple_returns(self, adj = False, cum = False):
        df = self.get_data()
        if adj:
            simple_returns = df["Adj Close"].pct_change().dropna()
        else:
            simple_returns = df["Close"].pct_change().dropna()
        if cum:
            simple_returns = (1 + simple_returns).cumprod() - 1
        return simple_returns
    
    def log_returns(self, adj = False, cum = False):
        simple_returns = self.simple_returns(adj, cum)
        log_returns = np.log(1+simple_returns)
        return log_returns
    
    
    def portfolio_returns(self, adj = False, cum = False):
        returns = self.simple_returns(adj, cum)
        portfolio_returns = (returns*self.weights).sum(axis=1)
        return portfolio_returns
    
    def returns_plot(self, port = False, adj = False, cum = False, log = False, template = 'plotly_dark'):
        returns = self.simple_returns(adj, cum).mul(100)
        if log:
            returns = self.log_returns(adj, cum).mul(100)
        if isinstance(returns, pd.Series): # This checks if the 
            returns.name = self.assets[0]
        fig = px.line(returns, template = template)
        fig.update_traces(hovertemplate='%{y}')
        fig.update_layout(
            legend = dict(title = None, font = dict(size = 16)),
            title={
                'y':0.95,
                'x':0.5,
                'text': "Asset Returns",
                'font': {'size': 24},
                'xanchor': 'center',
                'yanchor': 'top'},
            hovermode = "x unified",
            xaxis_title = "Date",
            yaxis_title = "% Returns")
        
        if port:
            portfolio_returns = self.portfolio_returns(adj, cum)
            fig = px.line(portfolio_returns, template = template)
            fig.update_traces(name = 'Returns',hovertemplate='%{y}')
            fig.update_layout(
                legend = dict(title = None, font = dict(size = 16)),
                title={
                    'y':0.95,
                    'x':0.5,
                    'text': "Portfolio Returns",
                    'font': {'size': 24},
                    'xanchor': 'center',
                    'yanchor': 'top'},
                hovermode = "x unified",
                xaxis_title = "Date",
                yaxis_title = "% Returns")
                
        fig.show() 
    
    def cov_matrix(self, plot = False, cum = False, template = 'plotly_dark'):
        returns = self.simple_returns()
        if cum:
            returns = self.simple_returns(cum)
        cov_matrix = returns.cov()
        
        if plot:
            fig = px.imshow(cov_matrix, text_auto=True, color_continuous_scale='tempo', template = template, title = "Covariance Matrix")
            fig.update_layout(
                legend = dict(title = None),
                title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
                )
            fig.show()
        return cov_matrix
        
    def corr_matrix(self, plot = False, cum = False, template = 'plotly_dark'):
        returns = self.simple_returns()
        if cum:
            returns = self.simple_returns(cum)
        corr_matrix = returns.corr()
        if plot:
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='tempo', template = template, title = "Correlation Matrix")
            fig.update_layout(
                legend = dict(title = None),
                title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
                )
            fig.show()
        return corr_matrix
    
        #Standard Deviation of Portfolio w/ Optional Crypto Arg
    def port_std(self, crypto = False):
        if crypto:
            trading_days = 365
        else:
            trading_days  = 252
        if len(self.assets) == 1:
            returns = self.simple_returns()
            summary = returns.describe().T.loc[["std"]]
            port_std  = round(summary["std"]*np.sqrt(trading_days),2)    
        else:
            cov_matrix = self.cov_matrix()
            port_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
            port_std = np.sqrt(port_variance) * np.sqrt(trading_days)
        

        return port_std
    
    def risk_return(self, adj = False, crypto = False, template = 'plotly_dark'):
        returns = self.simple_returns(adj).mul(100)
        if crypto:
            trading_days = 365
        else:
            trading_days  = 252
        summary = returns.describe().T.loc[:,["mean","std"]]
        summary["mean"] = round(summary["mean"]*trading_days,2) # Multiply by number of trading days
        summary["std"]  = round(summary["std"]*np.sqrt(trading_days),2)
        summary.rename(columns = {'mean':'% Return', 'std':'Risk'}, inplace = True) 
        fig = px.scatter(summary, 
                         x = 'Risk', 
                         y = '% Return', 
                         title = "Annual Risk / Return",
                         text = summary.index,
                         template = template)
        fig.update_traces(hovertemplate='Risk: %{x}<br>Return: %{y}')
        fig.update_traces(marker={'size': 15},
                          textposition='top center',
                          hoverlabel=dict(font=dict(size=15) ))
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
    
    def pie_plot(self, template = 'plotly_dark'):
        data = pd.DataFrame({"Assets": self.assets,
                             "Weights": self.weights})
        fig=go.Figure(go.Pie(labels=data['Assets'],
                                 values=data['Weights'],
                                 name = "",
                                 textinfo = 'label + percent'))
        fig.update_layout(template = template)
        fig.update_traces(hovertemplate='%{label}: %{percent}')
        fig.show()


class Asset:
    def __init__(self, ticker, start_date, end_date = datetime.today().strftime('%Y-%m-%d')):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date 
        
    def get_data(self):
        df = yf.download(self.ticker, start = self.start_date, end = self.end_date)
        return df
    
    def simple_returns(self, adj = False, cum = False):
        df = self.get_data()
        if adj:
            simple_returns = df["Adj Close"].pct_change().dropna()
        else:
            simple_returns = df["Close"].pct_change().dropna()
        if cum:
            simple_returns = (1 + simple_returns).cumprod() - 1
        return simple_returns
    
    def log_returns(self, adj = False, cum = False):
        simple_returns = self.simple_returns(adj, cum)
        log_returns = np.log(1+simple_returns)
        return log_returns
    
    def std(self, adj = False, crypto = False, template = 'plotly_dark'):
        returns = self.simple_returns(adj).mul(100)
        if crypto:
            trading_days = 365
        else:
            trading_days  = 252
        std = returns.describe().T.loc["std"]
        std = std*np.sqrt(trading_days)
        return std
   
    def mean_return(self, adj = False, crypto = False, template = 'plotly_dark'):
        returns = self.simple_returns(adj).mul(100)
        if crypto:
            trading_days = 365
        else:
            trading_days  = 252
        mean = returns.describe().T.loc["mean"]
        mean = mean*trading_days # Multiply by number of trading days 
        return mean
    
    def returns_plot(self, adj = False, cum = False, log = False, template = 'plotly_dark'):
        returns = self.simple_returns(adj, cum).mul(100)
        if log:
            returns = self.log_returns(adj, cum).mul(100)
        returns = returns.to_frame()
        returns = returns.rename(columns={'Close': 'Returns'})
        fig = px.line(returns, template = template)
        fig.update_traces(hovertemplate='%{y:.2f}%')
        fig.update_layout(
            showlegend = False,
            title={
                'y':0.95,
                'x':0.5,
                'text': f"{self.ticker} Daily Returns",
                'font': {'size': 24},
                'xanchor': 'center',
                'yanchor': 'top'},
            hovermode = "x unified",
            xaxis_title = "Date",
            yaxis_title = "% Returns")
                
        fig.show() 
        return returns
        
    def close_plot(self, adj = False, normalize = False, template = 'plotly_dark'):
        df = self.get_data()
        
        if adj:
            df["CLose"] = df["Adj Close"]
            title = f"{self.ticker} Adjusted Closing Price"
        else:
            title = f"{self.ticker} Closing Price"
            
        if normalize:
            df["Close"] = df["Close"].div(df["Close"].iloc[0]) #Normalizes data
            # normclose = normclose.to_frame()
            fig = px.line(df["Close"], 
                          x = df.index,
                          y = df["Close"],
                          title = "Normalized " + title,      
                          template = template) # Plotting Normalized closing data
            fig.update_traces(hovertemplate='Price: $%{y}')
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
            
            fig = px.line(df["Close"], 
                          x = df.index,
                          y = df["Close"],
                          title = title,
                          template = template) # Plotting Normalized closing data
            fig.update_traces(hovertemplate='Price: $%{y}')
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
                yaxis_title = "Closing Price (USD)"
                )
            fig.show()    
        
        
    def candlestick(self, sma1 = 0, sma2 = 0, template = 'plotly_dark'):
        
        ticker = Portfolio(self.ticker, self.start_date)
        df = ticker.get_data()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index,
                                      open=df['Open'],
                                      high=df['High'],
                                      low=df['Low'],
                                      close=df['Close'],
                                      name='candlestick'),
                                     
                      row=1, col=1)
        fig.update_traces(increasing_line_width = 1.5,
                          decreasing_line_width = 1.5
                         )
        if sma1 > 0:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=sma1).mean(),
                                      name=f'{sma1}-day moving average', line=dict(color='lightblue', width = 1)),
                          row=1, col=1)
        if sma2 > 0:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=sma2).mean(),
                                      name=f'{sma2}-day moving average', line=dict(color='red', width = 1)),
                          row=1, col=1)
        
        fig.add_trace(go.Bar(x=df.index,
                             y=df['Volume'],
                             name='volume'),
                      row=2, col=1)
        
        fig.update_layout(
                          title={
                          'text': f'{self.ticker} Candlestick Chart with Volume',
                          'y':0.9,
                          'x':0.5,
                          'font': {'size': 24},
                          'xanchor': 'center',
                          'yanchor': 'top',},
                          xaxis_rangeslider_visible=False,
                          xaxis_title='Date',
                          yaxis_title='Price (USD)',
                          hovermode = "x unified",
                          # xaxis_type="category",
                          bargap=0,
                          bargroupgap=0,
                          template = template)
        fig.update_xaxes(title_text='', row=1, col=1, showgrid=False)
        fig.update_xaxes(title='Date', row=2, col=1)
        fig.update_yaxes(title='Volume',row=2,col=1)
        
        fig.show()


spy = Asset(ticker, start_date)
sr = spy.returns_plot(cum = True)
# spy.candlestick(sma1 = 20, sma2 = 50)

# port = Portfolio(tickers, start_date)
# std = port.port_std()
# port.pie_plot()
# port.returns_plot(cum = True)
# port.close_plot(normalize = True)
# candlestick(ticker = 'SPY',start_date = '2020-01-01', sma1 = 10, sma2 = 50, template = 'plotly_dark')

