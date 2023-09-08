import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow 
from tensorflow import keras
import matplotlib.pyplot as plt
from io import BytesIO

data = pickle.load(open('original_df.pkl','rb'))
model = tensorflow.keras.models.load_model('gru_original_data.h5')

scl = StandardScaler()
scaled_data = scl.fit_transform(data)
std = scl.scale_
mean = scl.mean_

def prepare(data, n_features):
    X, y =[], []
    for i in range(len(data)):
        # find the last index
        end_ix = i + n_features
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
X, y = prepare(scaled_data, 30)
x_in = X[-1].reshape(1,30)
n_steps = 1
n_features = 30
x_input = np.array(x_in[0])

def forecasting(n_steps, n_features, x_input, model, period, num_of_pred):
    temp_input=list(x_input)
    lst_output=[]
    i=0
    while(i<num_of_pred):

        if(len(temp_input)>30):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i+1,x_input))
            #print(x_input)
            x_input = x_input.reshape((1, n_features, n_steps))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i+1,yhat))
            print(f'{i+1} {period} output {yhat}')
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.append(yhat[0][0])
            i=i+1
        else:
            x_input = x_input.reshape((1, n_features,n_steps))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i=i+1
    return lst_output

# Define the Streamlit app
def main():
    # Set the app title
    st.title('Demand Forecasting Model')

    # Create a selectbox widget
    start_date = '2017-08-16'
    st.write(f"Start Date: {start_date}")
    date_option = st.selectbox("Select End date: ",(pd.date_range(start='2017-08-16', end='2018-08-16').date))
    graph_option = st.radio("Select Plot:", ['line_chart','area_chart'])

    # Create a submit button
    if st.button("Submit"):

        end_date = date_option
        date_index = pd.date_range(start=start_date, end=end_date).date
        #num_of_pred = selected_option
        num_of_pred = len(pd.date_range(start=start_date, end=end_date).date)
        st.write(f"Number of Days Selected: {num_of_pred}")
        y_pred = forecasting(n_steps, n_features, x_input, model, 'day', num_of_pred)
        Forecasted = (y_pred * std) + mean

        if graph_option == 'line_chart':
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.plot(date_index,Forecasted)
            ax.set_xlabel("Day")
            ax.set_ylabel("Forecasted Value")
            ax.set_title("Demand Forecasting")
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig)

        elif graph_option == 'area_chart':
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.fill_between(date_index, Forecasted.reshape(len(Forecasted),), alpha=0.5)
            ax.set_xlabel("Day")
            ax.set_ylabel("Forecasted Value")
            ax.set_title("Demand Forecasting")
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig)
        # Save the chart as an image
        chart_image = BytesIO()
        plt.savefig(chart_image, format='png')
        plt.close()
        # Create a download button for the chart image
        st.download_button(
            label="Download Chart",
            data=chart_image.getvalue(),
            file_name='forecast_chart.png',
            mime='image/png',
        )
        Forecast = pd.DataFrame(Forecasted, index=pd.Index(name='Date', data=date_index),columns=['Sales'])
        st.dataframe(Forecast)
        # downloading files
        csv_data = Forecast.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name='sample_data.csv',
            mime='text/csv',
        )
        


if __name__ == "__main__":
    main()

