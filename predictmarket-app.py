import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import math




def main():

    df = load_data()

    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title("Stock Market Predictions with LSTM")
        st.subheader("Stok Market for SANOFI within the last 10 years")
        st.markdown("The goal of this project is to predict with the model LSTM the closing stock price of a corporation SANOFI using the the past 10 years.")
        image = Image.open("image_acceuil.PNG")
        st.image(image, use_column_width=True)
        st.title("Data from Yahoo's Finance API")
        st.dataframe(df)
        
    elif page == 'Exploration':
        st.title('Data Visualization')
        st.subheader('Mid Price history of SANOFI')
        plt.figure(figsize = (20,15))
        plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
        plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Mid Price',fontsize=18)
        st.pyplot()  
            
    else:
        st.title('Modelling')
        st.subheader('We are going to predict a closing price with LSTM network')
        st.subheader('RMSE')
        img1 = Image.open("RMSE.PNG")
        st.image(img1, use_column_width=True)
        st.subheader('Visualize the predicted stock price with original stock price')
        st.markdown("The exact price points from our predicted price is close to the actual price")
        img = Image.open("Result_final.PNG")
        st.image(img, use_column_width=True)
        st.subheader('Original stock price & Stock Price Prediction')
        img2 = Image.open("Data_predict.PNG")
        st.image(img2, use_column_width=True)

@st.cache
def load_data():
    return pd.read_csv("history_stock_Sanofi.csv")

if __name__ == '__main__':
    main()



