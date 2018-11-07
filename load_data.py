import os
import pandas as pd
import numpy as np
from DataTools.pickle import save_to_pickle

def check_dataframe_nans_dtype(df):
    ''' for pandas dataframes '''
    print('\ncheck nans and dtype\n')
    print('type: {}\n'.format(type(df)))
    print(df.isna().sum())

if __name__=='__main__':
    fname = 'household_power_consumption.txt'
    params = {'sep':';',
                'header':0,
                'low_memory':False,
                'infer_datetime_format':True,
                'parse_dates':{'datetime':[0,1]},
                'index_col':'datetime'}

    data = pd.read_csv(os.path.join(os.getcwd(),'data',fname), **params)
    # give a name
    data.name = 'data'
    print(data.name)
    print(data.shape)
    print('date range: {} to {}'.format(data.index[0], data.index[-1]))
    print(data.head())

    '''
    dataframe shape: (2075259, 9)
    sampling rate: 1 minute
    date range: 16/12/2006 to 26/11/2010

    dataframe columns:
    ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
           'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
           'Sub_metering_3']
    '''

    # replace missing value with nan
    data.replace('?', np.nan, inplace=True)
    # set data type as int
    data = data.astype('float32')

    # check nans and dtype
    check_dataframe_nans_dtype(data)

    # impute missing values
    from DataTools.impute import df_impute_previous_index
    import time
    tic = time.clock()
    data = df_impute_previous_index(data, 1, np.nan)
    toc = time.clock()
    print('time: {}'.format(toc-tic))

    # check again
    check_dataframe_nans_dtype(data)

    # downsample 
    data_30min = data.resample('30Min').sum() # must apply .mean() or .sum() to get df object
    check_dataframe_nans_dtype(data_30min)

    save_to_pickle(data, 'data', 'data_raw.pkl')
    save_to_pickle(data_30min, 'data', 'data_30min.pkl')
