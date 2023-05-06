import datetime
import streamlit as st
import yfinance as yf

@st.cache
def load_data():
    stock_data = yf.Ticker('MSFT')
    AAPL_stock_data = yf.Ticker('AAPL')
    return stock_data, AAPL_stock_data

stock_data, AAPL_stock_data = load_data()

st.write('# Financial App')  # Markdown

st.sidebar.subheader('Stock selection')
st.sidebar.write('Select your stock')
stocks = st.sidebar.multiselect('Select stock', ['AAPL', 'MSFT', 'AMZN', 'GOOGL'])


min_date = datetime.date(2021, 1, 1)
max_date = datetime.date.today()

(selected_min, selected_max) = st.sidebar.date_input('Pick date', (min_date, max_date))

ticker_dict = {}
for stock in stocks:

    stk = yf.Ticker(stock)
    historical_data = stk.history(period='1d', start=selected_min, end=selected_max)
    ticker_dict[stock] = historical_data.Close

st.line_chart(ticker_dict)
