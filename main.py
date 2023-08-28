import requests
import streamlit as st
from streamlit_lottie import st_lottie
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu


import yfinance as yf
import pandas as pd
import base64
import matplotlib.pyplot as plt 

from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.set_page_config(page_title="Stock Analyzer", page_icon="📊", layout="wide")

def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_MMj3wUKeHt.json")
lottie_people = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_qzasi9ko.json")
lottie_contact = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_px0ntw70.json")
lottie_graph = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_w9wl8mlm.json")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css") 
   
if True:    
    with st.container():
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Explore", "Stocks", "Crypto"],  # required
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
    if selected == "Home":
            with st.container():
                left_column, right_column = st.columns(2)
                
            with left_column:

                st.title("STOCK ANALIZE AND PREDICTIONS!📊")
                st.write("This site was created to group the stocks by companies in different sectors and also to predict future of stocks using machine learning.")
                

            with right_column:
                
                st_lottie(lottie_graph, height=300, key = "grpah")

            with st.container():
                left_column, right_column = st.columns(2)
                with left_column:
                    st.title("Information About Project")
                    st.write("##")
                    st.write(
                        """
                        - This Website is created using Streamlit python addon module for website design.
                        - This Website can show current live info about stock prices rise and fall.
                        - Using machine learing this site can also predict future price graphs of the stocks. 
                        """
                    )
                    st.write("All the stock live information is taken from [Yahoo Finance](https://finance.yahoo.com/)")
                with right_column:    
                    st_lottie(lottie_coding, height=300, key = "coding")

            with st.container():
                left_column, right_column = st.columns(2)

            with left_column:    
                st.write("---")
                st.header("Mini Project Made by...")
                st.write("""
                - Shreyas Patange  
                - Yash Pathare     
                - Mayuresh Parache 
                """)   

            with right_column:
                
                st_lottie(lottie_people, height=300, key = "people")

            contact_form = """
                    <form action="https://formsubmit.co/44a5b49c2f1fc754fa5a3eafecd973e9" method="POST">
                        <input type="hidden" name="_captcha" value="false">
                        <input type="text" name="name" placeholder="Your Name" required>
                        <input type="email" name="email" placeholder="Your Email" required>
                        <textarea name="Message" placeholder="Your message here.." required></textarea>
                        <button type="submit">Send</button>
                    </form>
                    """
            with st.container():
                st.write("---")
                st.header("Get in touch with us") 
                left_column, right_column = st.columns(2)
                
                
                with left_column:
                    st.write("##")
                    st.markdown(contact_form, unsafe_allow_html=True)
                
                with right_column:
                    st_lottie(lottie_contact, height=400, key = "contact")


            st.sidebar.image(
            "https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje",
            width=50,
            )
            st.sidebar.title(f"Welcome ")

    if selected == "Explore":
        st.title("S&P 500 Stock Price Explorer")

        # Description of the App
        st.markdown("""
        This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
        * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
        """)

        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")

        # Creating a Sidebar
        st.sidebar.image(
            "https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje",
            width=50,
        )
        st.sidebar.header('User Input Features')

        # Web scraping of S&P 500 data from Wikipedia
        @st.cache
        def load_data():
            # Take the url, from where you want to scrape the data
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            html = pd.read_html(url, header = 0)
            # html[0] means the first table from the website
            # We are storing that table as a Pandas Dataframe object
            df = html[0]
            return df

        # Call the Function and create the Dataframe
        df = load_data()

        # Examining the Sectors (There are mostly 11 sectors in the S&P500 data)
        # Get the Unique Sectors 
        sectors_unique = df['GICS Sector'].unique()

        # Aggregate the data
        sector = df.groupby('GICS Sector')

        # Sidebar - Sector Selection
        sorted_sector_unique = sorted(sectors_unique)

        # Get the selected sector
        selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

        # Filtering the data from the selected sectors in the sidebar
        df_selected_sector = df[(df['GICS Sector'].isin(selected_sector))]

        st.header('Display companies in the Selected Sector')
        st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
        st.dataframe(df_selected_sector)

        # Download S&P500 data in .csv format
        # This function creates a hyperlink in our web app to download the data
        def file_download(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
            return href

        st.markdown(file_download(df_selected_sector), unsafe_allow_html=True)

        # Let us use yfinance to download the stock data
        data = yf.download(
                tickers = list(df_selected_sector[:10].Symbol),
                period = "ytd",
                interval = "1d",
                group_by = 'ticker',
                auto_adjust = True,
                prepost = True,
                threads = True,
                proxy = None
            )
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Plot Closing Price of Query Symbol
        def price_plot(symbol):
            df = pd.DataFrame(data[symbol].Close)
            df['Date'] = df.index
            plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
            plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
            plt.xticks(rotation=90)
            plt.title(symbol, fontweight='bold')
            plt.xlabel('Date', fontweight='bold')
            plt.ylabel('Closing Price', fontweight='bold')
            return st.pyplot()

        num_company = st.sidebar.slider('Number of Companies', 1, 5)

        if st.button('Show Plots'):
            st.header('Stock Closing Price')
            for i in list(df_selected_sector.Symbol)[:num_company]:
                price_plot(i) 
    
    if selected == "Stocks":
        st.sidebar.image(
            "https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje",
            width=50,
        )
        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        st.title('Stock Forecast App')

        stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'GC=F')
        selected_stock = st.selectbox('Select dataset for prediction', stocks)

        n_years = st.sidebar.slider('Years of prediction:', 1, 4)
        period = n_years * 365


        @st.cache
        def load_data(ticker):
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data

            
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())
            
        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)              
    
    if selected == "Crypto":
        st.title("Crypto Prices")
        st.sidebar.image(
            "https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje",
            width=50,
        )

        c1, c2 = st.columns([1, 8])

        with c1:
            st.image(
                "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/chart-increasing_1f4c8.png",
                width=90,
            )

        st.markdown(
            """ **Crypto Dashboard**
        A simple cryptocurrency price app pulling price data from the [Binance API](https://www.binance.com/en/support/faq/360002502072).
            """
        )

        st.header("**Selected Price**")

        # Load market data from Binance API
        df = pd.read_json("https://api.binance.com/api/v3/ticker/24hr")

        # Custom function for rounding values
        def round_value(input_value):
            if input_value.values > 1:
                a = float(round(input_value, 2))
            else:
                a = float(round(input_value, 8))
            return a


        crpytoList = {
            "Price 1": "BTCBUSD",
            "Price 2": "ETHBUSD",
            "Price 3": "BNBBUSD",
            "Price 4": "XRPBUSD",
            "Price 5": "ADABUSD",
            "Price 6": "DOGEBUSD",
            "Price 7": "SHIBBUSD",
            "Price 8": "DOTBUSD",
            "Price 9": "MATICBUSD",
        }

        col1, col2, col3 = st.columns(3)

        for i in range(len(crpytoList.keys())):
            selected_crypto_label = list(crpytoList.keys())[i]
            selected_crypto_index = list(df.symbol).index(crpytoList[selected_crypto_label])
            selected_crypto = st.sidebar.selectbox(
                selected_crypto_label, df.symbol, selected_crypto_index, key=str(i)
            )
            col_df = df[df.symbol == selected_crypto]
            col_price = round_value(col_df.weightedAvgPrice)
            col_percent = f"{float(col_df.priceChangePercent)}%"
            if i < 3:
                with col1:
                    st.metric(selected_crypto, col_price, col_percent)
            if 2 < i < 6:
                with col2:
                    st.metric(selected_crypto, col_price, col_percent)
            if i > 5:
                with col3:
                    st.metric(selected_crypto, col_price, col_percent)

        st.header("")


        @st.cache
        def convert_df(df):
            return df.to_csv().encode("utf-8")


        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="large_df.csv",
            mime="text/csv",
        )

        st.dataframe(df, height=2000)


        st.markdown(
            """
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        """,
            unsafe_allow_html=True,
        )
