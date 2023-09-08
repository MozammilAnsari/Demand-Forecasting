import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow 
from tensorflow import keras
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
    selected_option = st.selectbox("Select Number of Days:", range(1,365))
    graph_option = st.radio("Select Plot:", ['line_chart','area_chart'])
    # Create a submit button
    if st.button("Submit"):
        # Do something when the button is clicked, e.g., display the selected option
        st.write(f"Selected Number of Days: {selected_option}")
        
        num_of_pred = selected_option
        y_pred = forecasting(n_steps, n_features, x_input, model, 'day', num_of_pred)
        Forecasted = pd.DataFrame((y_pred * std) + mean)
        if graph_option=='line_chart':
            st.line_chart(Forecasted)
        elif graph_option=='area_chart':
            st.area_chart(Forecasted)
        st.dataframe(Forecasted)
        # downloading files
        csv_data = Forecasted.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name='sample_data.csv',
            mime='text/csv',
        )
        


if __name__ == "__main__":
    main()
