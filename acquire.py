import pandas as pd
import numpy as np
import os


#################### Acquire Pick Data ##################

def get_data():
    '''
    This function reads in text file and returns df
    '''
    # assign column names for incoming file
    colnames=['date', 'hour', 'page_viewed', 'user_id', 'cohort_id', 'IP']
    df = pd.read_csv('anonymized-curriculum-access.txt', sep=' ', header=None, names=colnames)
    return df

def get_cohorts():
    '''
    This function reads in text file and returns df
    '''
    cohorts = pd.read_csv('cohorts.csv', index_col=0)
    return cohorts

def run():
    print("Acquire: downloading raw data files...")
    df = get_data()
    cohorts = get_cohorts()
    # merge these into one dataframe
    df = df.merge(cohorts, how='left', on='cohort_id')  
    print("Acquire: Completed!")
    return df
