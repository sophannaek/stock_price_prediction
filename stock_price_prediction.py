import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf
import math


# prediction based on the last 60days price 
# [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo] for interval 

######### functions #############
@st.cache
def getSP500Symbol():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

# display the FAANG companies performances
def displayFAANG(): 
    faangSymbols = ['AAPL','AMZN', 'GOOGL','FB' ,'NFLX']
    title = "FAANG Stock Prices Since January 2021"
    return multiple_plots(faangSymbols, 'ytd','1d', title)

def multiple_plots(selected_company, selected_period, selected_interval, title):
    data = downloadData(selected_company, selected_period,selected_interval)
    colors = ['skyblue', 'red', 'purple', 'blue', 'pink']
    i  = 0
    if not title: 
        title="Stock Prices since January 2021"
    for sym in selected_company: 
        plt.plot(data[sym].Close, color=colors[i])
        plt.xticks(rotation=90)
        plt.legend(selected_company)
        plt.title(title, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        i += 1
        # limited to 5 plots for better visualization purpose
        if i == 5: 
            break
    st.pyplot(plt)


# get the data 
def downloadData(companies, selected_period, selected_interval):
    data = yf.download(
        # tickers = list(companies),
        tickers = companies,
        period = selected_period,
        interval = selected_interval, 
        group_by = 'ticker',
        auto_adjust = True, 
        prepost = True, 
        threads = True, 
        proxy = None
    )
    return data

# term: 30, 45, 60, 90 days 
def getTrainingSet(data, term):
    df = pd.DataFrame(data, columns = ['Open','High', 'Low', 'Close','Volume'])        
    df['Date'] = df.index
    # convert dataframe to a numpy array 
    dataset = df.filter(['Close'])
    # #Get the number of rows to train the model on 
    training_data_len = math.ceil(len(dataset) * 0.80)

    # #scale the data 
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    x_train=[]
    y_train=[]

    # # training data 
    train_data = scaled_data[0:training_data_len, :]
    for i in range(term, len(train_data)):
        # each pass contains term (30, 60 or 90 ) values
        x_train.append(train_data[i-term:i,0])
        y_train.append(train_data[i,0])
    # Convert the x_trian and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)
    #reshape the data -- input data for LSTM must be 3D (number of sample data, timestep and features)
    print(x_train.shape)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    # x_train.shape

    # create the testing data set
    test_data = scaled_data[training_data_len - term: , :]
    #create a data sets x_test and y_test
    x_test = []
    y_test = []
    # test_data = dataset[training_data_len:, :]
    for i in range(term, len(test_data)):
        x_test.append(test_data[i-term:i, 0])
        y_test.append(test_data[i,0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

    return x_train, y_train, x_test, y_test , training_data_len, scaler


def build_model(x_train,y_train,x_test,y_test, scaler): 
    # Build the LTSM model
    model = Sequential()
    # input neural network layer
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    # hidden neural network layers 
    model.add(LSTM(55, return_sequences=False))
    model.add(Dense(15))
    #ouput neural network layer 
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=2, epochs = 5)

    # prediction
    predictions = model.predict(x_test)
    # scale the data back 
    predictions = scaler.inverse_transform(predictions)
    # get the error rate: RMSE 
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    
    return rmse, predictions, model

def graph_prediction(data, prediction1, prediction2, training_data_len):
    # plot the prediction 
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Prediction1'] = prediction1
    valid['Prediction2'] = prediction2
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Prediction1','Prediction2']])
    # plt.plot(data['Close'])
    # plt.plot(valid[['Prediction1','Prediction2']])
    plt.xticks(rotation=90)
    plt.legend(['TrainingSet Price', 'True Price', 'Predicted Price - 60days','Predicted Price - 90days'], loc='lower right')
    # plt.legend(['Actual Price', 'Predicted Price - 60days','Predicted Price - 90days'], loc='lower right')

    st.pyplot(plt)

    

def make_prediction(selected_prediction,term):
    data = downloadData(selected_prediction, '2y','1d')
     # prep training set 
    x_train, y_train, x_test, y_test,training_data_len , scaler = getTrainingSet(data,term)
    # build the model 
    rmse, predictions, model = build_model(x_train, y_train, x_test, y_test, scaler)
     
    df = pd.DataFrame(data,columns = ['Open','High', 'Low', 'Close','Volume'] ) 
    df['Date'] = df.index
    df = data.filter(['Close'])

    last_term_price = df[-term:].values
    scaled = scaler.transform(last_term_price)

    # prep the test data for prediction
    x_test = []
    x_test.append(scaled)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    # make the preidction 
    pred = model.predict(x_test)
    pred = scaler.inverse_transform(pred)

    return predictions, pred.item(0), training_data_len, data



def keyStatistics(df): 
    volume = round(float(df.iloc[-1]['Volume']/1000000),2)
    avg_volume = round(df['Volume'].mean()/1000000,2)
    maxPrice = round(df['Close'].max(),2)
    minPrice = round(df['Close'].min(),2)

    st.header('Key Statistics: ')        
    left_column, right_column = st.beta_columns(2)
    left_column.write("**High Today:  ** $" + str(round(df['High'].iloc[-1],2)))
    left_column.write("**Open Price:  ** $" + str(round(df.iloc[-1]['Open'],2)))
    right_column.write("**Low Today:  ** $"+ str(round(df.iloc[-1]['Low'],2)))
    right_column.write("**Close Price:  ** $"+ str(round(df.iloc[-1]['Close'],2)))
    left_column.write("**Volume:  **"+ str(volume)+ " M")
    right_column.write("**Average Volume: **"+ str(avg_volume) + " M")
    left_column.write("**YTD High: ** $" + str(maxPrice))
    right_column.write("**YTD Low: ** $ "+ str(minPrice))



######################
sp_list = getSP500Symbol()
symbols = sp_list['Symbol']
# prediction_symbols = list(symbols)
prediction_symbols = symbols
prediction_symbols = symbols.loc[len(symbols.index)]=''
print(symbols)
print(prediction_symbols)

st.title('Welcome to the S&P500 Stock Market')



######################### side bar ##############
# # Sidebar - Company selection
st.sidebar.header("Company's Performance")
selected_company = st.sidebar.multiselect(' Select a company or multiple companies to see their perfomances',
     sorted(symbols))


if selected_company: 
    # selected_prediction= st.empty()
    # selected_prediction.selectbox("Choose your company", sorted(symbols))
    period = ['1y', '3m', '6m', '1w','2y']
    selected_period = st.sidebar.selectbox('Select the period you\'d like to view!',
        period
    )

    interval = ['1d','1w']
    selected_interval = st.sidebar.selectbox('Select the interval', interval)


# stock price prediction 
st.sidebar.header("Stock Price Prediction")
selected_prediction = st.sidebar.selectbox("Choose your company",    sorted(symbols))

print("selected_prediction ", selected_prediction)
if(selected_prediction):
    st.write("You select **" , selected_prediction ,"** for price prediction. Click **Predict Now?** to start now!")

if selected_prediction:
    pressed = st.sidebar.button('Predict Now?')
    if pressed: 
        
        gif_runner = st.image("https://media.giphy.com/media/kUTME7ABmhYg5J3psM/giphy.gif")
        pred_price_60days = make_prediction(selected_prediction, 60)
        pred_price_90days = make_prediction(selected_prediction, 90)
        gif_runner.empty()
        st.write("The predicted price for **",selected_prediction,"** based on the last 60 days performance: $", round(pred_price_60days[1],2))
        st.write("The predicted price for **",selected_prediction,"** based on the last 90 days performance: $", round(pred_price_90days[1],2))
        graph_prediction(pred_price_60days[3], pred_price_60days[0], pred_price_90days[0], pred_price_60days[2])
        


######## Main Layout ##############



if selected_company and not selected_prediction: 
    data = downloadData(selected_company, selected_period, selected_interval)

    # if only one company, display a chart and its details 
    # if it is going down --> red , if it is going up --> blue or green 
    if len(selected_company) == 1:
        symbol = selected_company[0]
        print("selected company : ", selected_company)
        print(symbol)
        df = pd.DataFrame(data, columns = ['Open','High', 'Low', 'Close','Volume'])        
        df['Date'] = df.index
        plt.plot(data['Close'], color='skyblue')
        plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
        plt.fill_between(df.Date, df['Close'], color='skyblue', alpha=0.3)
        plt.xticks(rotation=90)
        plt.title(selected_company[0], fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Closing Price', fontweight='bold')
        st.pyplot(plt)

        keyStatistics(df)


    
    else: 
        print(selected_company)
        multiple_plots(selected_company,'ytd','1d','')

        # # if more than one selected companies print the data in a table 
        # for i in selected_company: 
        #     st.write(i)

elif not selected_prediction: 
    # st.write("default view")
    displayFAANG()







