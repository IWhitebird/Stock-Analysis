#home
import imp
from pickle import NONE
import pickle
from pathlib import Path
from select import select
from typing import Container
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu

#info
import yfinance as yf
import pandas as pd
import base64
import matplotlib.pyplot as plt

#predictions
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.set_page_config(page_title="Stock Analizer", page_icon = ":bar_chart:", layout = "wide")

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
       
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:    
    with st.container():
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Info", "Prediction"],  # required
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
                    st.header("Information About Project")
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
                st.title("Mini Project Made by..")
                st.write("""
                - Shreyas Patange  )
                - Yash Pathare     )
                - Mayuresh Parache )
                - Sudhanshu Prasad )
                """)   

            with right_column:
                
                st_lottie(lottie_people, height=300, key = "people")

            contact_form = """
                    <form action="https://formsubmit.co/evilzgaming2003@email.com" method="POST">
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



            st.sidebar.title(f"Welcome {name}")
            authenticator.logout("Logout", "sidebar") 

    if selected == "Info":
        st.title("S&P 500 Stock Price Explorer")

        # Description of the App
        st.markdown("""
        This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
        * **Python libraries:** base64, pandas, streamlit, matplotlib, yfinance
        * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
        """)

        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")

        # Creating a Sidebar
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
    
    if selected == "Prediction":
        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        st.title('Stock Forecast App')

        stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
        selected_stock = st.selectbox('Select dataset for prediction', stocks)

        n_years = st.slider('Years of prediction:', 1, 4)
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