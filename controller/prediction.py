# import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import itertools
import math
from sklearn.preprocessing import MinMaxScaler
from itertools import chain

### Tensorflow libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# fb prophet libraries
# from fbprophet import Prophet


SP_STOCK_CODE = '^GSPC'
STOCK_NAME_CODE_MAP = {
    '^GSPC': 'S&P 500',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'AMT': 'American Tower Corporation',
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'SHW': 'The Sherwin-Williams Company',
    'DD': 'DuPont de Nemours, Inc.',
    'BA': 'The Boeing Company',
    'UNP': 'Union Pacific Corporation',
    'DUK': 'Duke Energy Corporation',
    'ED': 'Consolidated Edison, Inc.',
    'AEP': 'American Electric Power Company, Inc.',
    'UNH': 'UnitedHealth Group Incorporated',
    'JNJ': 'Johnson & Johnson',
    'BRK-A': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'MCD': 'McDonald\'s Corporation',
    'KO': 'The Coca-Cola Company',
    'PG': 'The Procter & Gamble Company',
    'MSFT': 'Microsoft Corporation',
    'SPG': 'Simon Property Group, Inc.'
}
# STOCKS_CURRENT_PRICE = {}
STOCKS_CURRENT_PRICE = {'^GSPC': 4392.59, 'AAPL': 165.29, 'AMZN': 3034.13, 'AMT': 255.54, 'XOM': 87.83, 'CVX': 171.59, 'SHW': 252.9, 'DD': 68.7, 'BA': 181.94, 'UNP': 246.21, 'DUK': 114.85, 'ED': 98.03, 'AEP': 102.04, 'UNH': 534.82, 'JNJ': 179.9, 'BRK-A': 516435, 'JPM': 126.12, 'MCD': 250.51, 'KO': 65.02, 'PG': 158.57, 'MSFT': 279.83, 'SPG': 127.88}
STOCK_CODES = ' '.join(list(STOCK_NAME_CODE_MAP.keys()))
START_DATE = '1997-05-24' #From this date, yahoo finance has the data listed
END_DATE = str(datetime.datetime.today()).split()[0] # current date

SCALER = None
DF = None
TEST_DF = None
TEST_DATA = None
BETA_VALUES = None
TIME_STEPS = 100

MODEL = load_model("./models/NEW^GSPC.h5")



'''
  This function will download yahoo finance data.
  params:
  stockCode: (str) - code of the particular stock to uniquely identify it in yahoo finance
  startDate: (str[yyyy-mm-dd]) - starting date from which the data should be download
  endDate: (str[yyyy-mm-dd]) - ending date from which the data should be download
'''
def download_data(stockCode, startDate, endDate):
  # Read data 
  datasetDF = yf.download(stockCode,startDate,endDate)
  # print(datasetDF)
  datasetDF = datasetDF["Adj Close"]
  # print("---------original data-----------")
  # print(datasetDF)
  return datasetDF


'''
This function will clean null values and replace it with appropriate values
df - dataframe with all the stock data
'''
def clean_null_values(df):
  stockCodes = list(df.columns.values)
  
  for stockCode in stockCodes:
    if(df[stockCode].isnull().values.any()):
      # clean particular stocks null values
      df = df.fillna(df[stockCode].median())
  
  return df


def pre_process_data(df):
  # check if any null values present
  if(df.isnull().values.any()):
    df_clean = clean_null_values(df)
  else:
    df_clean = df
  
  return df_clean


'''
This function will return beta value of all the stocks in comparision of S&P 500
'''
def get_beta_values(df, SP_STOCK_CODE):
  # find log returns for each stocks (log return calculate percentage difference of each stock's current day to it's previous day)
  df_log_returns = np.log(df/df.shift())
  df_per_change = df_log_returns.pct_change()
  df_cov = df_log_returns.cov()

  SP_var = df_log_returns[SP_STOCK_CODE].var()
  beta_values = df_cov.loc[SP_STOCK_CODE]/SP_var

  return beta_values


def split_dateset(df):
  training_size=int(len(df)*0.90)
  test_size=len(df)-training_size

  train_data = df[0:training_size,:]
  test_data = df[training_size:len(df),:1]
  return train_data, test_data


def download_stocks_current_price():
  global STOCKS_CURRENT_PRICE
  for stock_code in STOCK_NAME_CODE_MAP.keys():
    print('getting price for stock: ', stock_code)
    STOCKS_CURRENT_PRICE[stock_code] = yf.Ticker(stock_code).info['regularMarketPrice']
  
  print(STOCKS_CURRENT_PRICE)


def get_test_data():
    global TIME_STEPS, SP_STOCK_CODE, START_DATE, END_DATE, SCALER, TEST_DATA, TEST_DF
    #download S&P data
    df = download_data(SP_STOCK_CODE, START_DATE, END_DATE)

    df = pre_process_data(df)
    # df = df.rolling(TIME_STEPS).mean()[TIME_STEPS-1:]

    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
    SCALER=MinMaxScaler(feature_range=(0,1))
    df = SCALER.fit_transform(np.array(df).reshape(-1,1))

    train_data, test_data = split_dateset(df)

    TEST_DATA = test_data
    TEST_DF = df

    # print("------------train data-----------")
    # print(train_data)
    # print("--------------------test data--------")
    # print(test_data)
    # return test_data


def initialize():
    print('initialization started')
    global DF, STOCK_NAME_CODE_MAP, SP_STOCK_CODE, BETA_VALUES, STOCK_CODES, START_DATE, END_DATE, TEST_DATA

    #1 Download data from yahoo finance
    df = download_data(STOCK_CODES, START_DATE, END_DATE)

    # download_stocks_current_price()

    #2 Pre-process data
    df = pre_process_data(df)
    
    # store df to global
    DF = df
    # print("-----------DF after pre-processing-----------")
    # print(DF)

    #3 find beta value of each stocks
    BETA_VALUES = get_beta_values(df, SP_STOCK_CODE)

    # print("-----------------df after beta values----------")
    # print(df)

    get_test_data()
    #4 store test data for prediction
    # TEST_DATA = get_test_data()
    # print('global test data length: ', len(TEST_DATA))



def generateRandomInputForModel():
    return np.random.random(100).reshape(1,100,1)


def filter_stocks_on_risk(stocksOnRiskLevel, riskLevel):
  totalRisksCount = 3

  maxStocksAfterFilter = None
  if(len(stocksOnRiskLevel.keys()) % 3 == 0):
    maxStocksAfterFilter = len(stocksOnRiskLevel.keys())//3
  else:
    maxStocksAfterFilter = len(stocksOnRiskLevel.keys())//3 + 1

  startIndex = max((riskLevel-1) * maxStocksAfterFilter, 0)
  stopIndex = min(riskLevel * maxStocksAfterFilter, len(stocksOnRiskLevel))
  return dict(itertools.islice(stocksOnRiskLevel.items(), startIndex, stopIndex))


def getStocksBasedOnRisk(inp):
    global DF
    df = DF
    # money = inp.investmentMoney
    # riskLevel = inp.riskLevel
    # money = 1000 # int(input('How much money you would like to invest: '))
    # # risk level
    # riskLevelMap = {
    #     "low": 1,
    #     "medium": 2,
    #     "high": 3
    # }
    # riskLevel = 'medium' #input(('What risk level you would prefer [High, medium, low]. Hint: High risk increase the changes of return, but also increase the chances of risk: '))
    # riskLevelInt = riskLevelMap.get(riskLevel.lower().strip(), None)

    money = inp['investmentMoney']
    riskLevelInt = inp['riskLevel']

    if( riskLevelInt and money):
        #1 sort stocks according to risk
        stocksOnRiskLevel = dict(sorted(dict(df.std()).items(), key=lambda x: x[1]))

        #2 get stocks based on risk level
        filteredStocks = filter_stocks_on_risk(stocksOnRiskLevel, riskLevelInt)
        # print(filteredStocks)

        #3 find correlation of filtered stocks
        diversificationCorr = df[list(filteredStocks.keys())].corr()

        flattened = diversificationCorr.unstack()
        sort_by_corr = flattened.sort_values(kind="quicksort")
        dict_sorted_corr = dict(sort_by_corr)

        n = round(math.log(money,10))
        list_sorted_corr = list(set([item for t in list(dict_sorted_corr.keys()) for item in t]))[0:n]
        # print(list_sorted_corr)
        return list_sorted_corr

initialize()


def get_stocks_future_data(stock_code, predicted_data):
  predicted_data_df = pd.DataFrame(predicted_data)
  daily_change = predicted_data_df.pct_change().values.tolist()[1:]
  daily_change_lst = [i[0] for i in daily_change]

  price = STOCKS_CURRENT_PRICE[stock_code]
  beta_value = BETA_VALUES[stock_code]

  stock_predicted_values = []
  for daily in daily_change_lst:
    stock_predicted_values.append(price + (daily * beta_value * price * 100))
  
  return stock_predicted_values


def predictModel(inp):
    global TEST_DATA, TIME_STEPS, TEST_DF, MODEL, SCALER, SP_STOCK_CODE
    df = TEST_DF
    model = MODEL
    scaler = SCALER
    userData = {
        'investmentMoney': inp['investmentMoney'],
        'riskLevel': inp['riskLevel'],
        'stock': inp['userSelectedStock'],
        'daysOfPrediction': inp['daysOfPrediction']
    }

    x_input = TEST_DATA[len(TEST_DATA)-TIME_STEPS:].reshape(1, -1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    i=0
    while(i<userData['daysOfPrediction']):
        if(len(temp_input)>TIME_STEPS):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, TIME_STEPS, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, TIME_STEPS,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    # final_output = list(chain.from_iterable(lst_output))
    day_new=np.arange(1, TIME_STEPS + 1)
    day_pred=np.arange(TIME_STEPS + 1, TIME_STEPS + userData['daysOfPrediction'] + 1)

    stock_predicted_values = get_stocks_future_data(userData['stock'], lst_output)
    print(stock_predicted_values)
    returnObj = {
        "original_data": DF[userData['stock']].reset_index().values.tolist(),
        "previous_days": day_new,
        "previous_days_data": list(chain.from_iterable(scaler.inverse_transform(df[len(df)-TIME_STEPS:]))),
        "predicted_days": day_pred,
        "predicted_days_data": stock_predicted_values #list(chain.from_iterable(scaler.inverse_transform(lst_output)))
    }
    # print(returnObj)
    return returnObj
    # print(returnObj['previous_days_data'])
    # print(df)
    # try:
    #     return model.predict(inp)[0][0]
    # except :
    #     return "System Error: Could not predict"



def predictFromFB(inp):
  userData = {
      'investmentMoney': inp['investmentMoney'],
      'riskLevel': inp['riskLevel'],
      'stock': inp['userSelectedStock'],
      'daysOfPrediction': inp['daysOfPrediction']
  }

  df = download_data(SP_STOCK_CODE, START_DATE, END_DATE)
  df = pre_process_data(df)
  print(len(df))

  fbp = Prophet(daily_seasonality=True)
  fbp.fit(df)
  future = fbp.make_future_dataframe(periods=365)
  forecast = fbp.predict(future)
  print(len(forecast))