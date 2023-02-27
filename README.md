# stock-analysis
This repository is for a project on stock analysis using python.  The python script contains three functions to help you analyze stocks.  They all require an input of a dataframe. While this dataframe doesn't technically have to be built from the yfinance.download() function, it must have the same structure and column names.  These functions were designed based around that function from the yahoo finance library.

The first function: close_plot() plots the closing prices of one or more stocks.  The function contains a few arguments: close_plot(df, adj = True, normalize = False).

"df" is the dataframe input required.  As I previously said, this dataframe is built from the yfinance.download() function, so it expects that format as input.
The dataframe input has a date index, and the following columns: "Adj Close", "Close", "High", "Low", "Open", and "Volume".  

"adj = True" refers to the "Adj Close" column.  This argument assumes that the user wants to use the adjusted closing price "Adj Close" rather than the closing price "Close" for plotting.  Setting "adj = False" will make the function use the "Close" column data for plotting.

"normalize = False" assumes that the user does not want to normalize the price data for plotting.  Setting this equal to true will divide all values for each closing price by the first closing price for that stock.  This allows the user to better compare stock performance of stocks with large differences in price.

