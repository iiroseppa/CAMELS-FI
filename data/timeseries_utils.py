import pandas as pd
import numpy as np

def read_daily_timeseries_csv(path, time_column='date', **kwargs):
    """Convenience function for reading a daily timeseries from csv, and setting index as datetimeindex.
    Kwargs are passed to pandas.read_csv
    """
    df = pd.read_csv(path, **kwargs)
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column)
    return df