import pandas as pd
import numpy as np
import scipy as sp 
import os
# from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#######  Pick Data Prep #######
def operator_outliers(df):
    '''
    remove outliers in operator, create variable to hold list of removed values
    '''
    op_list = df.operator.value_counts()
    operdf = pd.DataFrame(op_list)
    operdf.reset_index()
    operdf = operdf.rename(columns={'index': 'name', 'operator': 'occur_times'})
    one_off = operdf[operdf.occur_times < 10].index
    keep_op_list = operdf[operdf.occur_times > 9].index
    # keep only operators with more than 9 occurances 
    df = df[df.operator.isin(keep_op_list)]
    return df

def prep_data(df):
    '''
    Takes the acquired pick data, does data prep, and returns full df for exploration.
    No data split for modeling at this time.
    '''
    # merge date and hour together with a space between for later datetime conversion
    df['timestamp'] = df['date'] + ' ' + df['hour']
    # drop redundant cohort columns
    del df['date']
    del df['hour']

    # almost 45,000 records missing cohort id, and 1 page_viewed, fill all with 0 for now
    df = df.fillna(0)
    # drop null values from join and 1 page_viewed null
    #df = df.dropna()
    # change to integer instead of float
    df.cohort_id = df.cohort_id.astype('int')
    #df.user_id = df.user_id.astype('int')
    # convert the date_time to a datetime type
    df.timestamp = pd.to_datetime(df.timestamp) 
    df = df.set_index('timestamp')

    # add these columns for exploring
    df['hour'] = df.index.hour
    df['weekday'] = df.index.day_name()
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # No outlier removal yet
    # # run operator outlier removal before split because removal is based on domain knowledge
    # df = operator_outliers(df)
    # # drop negative pick seconds value
    # outlier2 = df[df.pick_seconds < 0].index
    # df = df.drop(outlier2)
    # # drop 7 observations where total_boxes <1
    # outlier3 = df[df.total_boxes < 1].index
    # df = df.drop(outlier3)
    # # run data outlier removal before split based on domain knowledge (want only 3 years with consistent range of volume)
    # df = df[(~(df['start'] < '2016-01-01')) & (~(df['start'] > '2019-12-31'))]

    return df 

#### NOTE: call the above with: train, test, validate = prep_pick_data(df)

def run(df):
    print("Prepare: Cleaning acquired data...")
    df = prep_data(df)
        
    # create df with non-time series index as well
    df_not_ts = df.copy()
    df_not_ts = df_not_ts.reset_index()

    print("Prepare: Completed! Data not split because no modeling at this time. Full dataset returned.")
    return df, df_not_ts
