import numpy as np
import os
import csv
import requests
import pandas as pd
import time
import datetime
from stockstats import StockDataFrame as Sdf
from ta import add_all_ta_features
from ta.utils import dropna
from ta import add_all_ta_features
from ta.utils import dropna

from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df

def load_stocks_data(stocks_data_file):
    url = 'https://finfo-api.vndirect.com.vn/v4/stocks?q=type:STOCK~status:LISTED&fields=code,type,floor,isin,status,companyName,companyNameEng,shortName,listedDate,indexCode,industryName&size=3000'
    
    print('retriving data from {}'.format(url))
    response = requests.get(url=url)
    data = response.json() 
    
    stocks_data = data['data']
    print('got stocks data with {} elements'.format(len(stocks_data)))

    stocks_df = pd.DataFrame(stocks_data)
    stocks_df.to_csv(stocks_data_file, index=False, encoding='utf-8')
    print('saved stocks data to {}'.format(stocks_data_file))

def to_timestamp(date_str):
    timestanp = int(time.mktime(datetime.datetime.strptime(date_str, "%d/%m/%Y").timetuple()))
    return timestanp

def to_date_str(timestamp):
    date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y%m%d")
    return date

def get_stock_price_part(stock_code, start_date, end_date):
    start_time = to_timestamp(start_date)
    end_time = to_timestamp(end_date)
    params = {
        "resolution": 'D',
        "symbol": stock_code,
        "from": start_time,
        "to": end_time
    }
    url = 'https://dchart-api.vndirect.com.vn/dchart/history'

    print('retreving price history for {} period {} - {}'.format(stock_code, start_date, end_date))
    response = requests.get(url=url, params=params)
    data = response.json()

    columns = {
        "tic": stock_code,
        "datadate": data["t"],
        "adjcp": data["c"],
        "close": data["c"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "volume": data["v"],
    }

    df = pd.DataFrame(columns)

    df['datadate'] = df['datadate'].astype(int)
    df['volume'] = df['volume'].astype(int)
    df['datadate'] = df['datadate'].apply(to_date_str)

    return df

def get_stock_price_history(stock_code):
    periods = [
        ('01/01/2001', '31/12/2006'),
        ('01/01/2007', '31/12/2012'),
        ('01/01/2013', '31/12/2008'),
        ('01/01/2019', datetime.datetime.today().strftime('%d/%m/%Y')),
    ]
    full_df = pd.DataFrame([])
    for period in periods:
        start_date, end_date = period
        period_df =  get_stock_price_part(stock_code, start_date, end_date)
        full_df = pd.concat([full_df, period_df])

    return full_df

def load_trading_data(stocks_data_file, training_data_file):
    print('load stocks data from {}'.format(stocks_data_file))
    price_df = pd.DataFrame([])
    stocks_df = pd.read_csv(stocks_data_file)

    qualified_stocks_df = stocks_df.loc[stocks_df['indexCode'] == 'VN30']

    for index, row in qualified_stocks_df.iterrows():
        stock_code = row['code']
        print('{}/{} load stock data {}'.format(index, qualified_stocks_df.size,  stock_code))
        stock_df = get_stock_price_history(stock_code)
        price_df = pd.concat([ price_df, stock_df])
    
    price_df.to_csv(training_data_file, index=False, encoding='utf-8')

def preprocess_data():
    """data preprocessing pipeline"""
    stocks_data_file = config.STOCKS_DATA_FILE
    training_data_file = config.TRAINING_DATA_FILE

    if not os.path.exists(stocks_data_file):
        load_stocks_data(stocks_data_file)
    
    if not os.path.exists(training_data_file):
        load_trading_data(stocks_data_file, training_data_file)

    df = load_dataset(file_name=training_data_file)
    # get data after 2009
    # df = df[df.datadate>=20090000]
    # calcualte adjusted price
    # df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)

    # df = dropna(df)

    # df_final = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")

    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index









