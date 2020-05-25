import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as pdr
from pandas_datareader import data
from datetime import datetime
from dater import Dater
import matplotlib.pyplot as plt
import os


def stocks(dates, mkt):
    start = datetime.strptime(dates[0], '%Y-%m-%d').date()
    end = datetime.strptime(dates[-1], '%Y-%m-%d').date()
    df = pd.DataFrame(data.get_data_yahoo('^GSPC', start=start, end=end))

    if not os.path.isfile('{}.csv'.format(mkt)):
        df.to_csv('{}.csv'.format(mkt), header=True)
    else:  # else it exists so append without writing the header
        df_old = pd.read_csv('{}.csv'.format(mkt))
        df1 = df_old.iloc[1:].rename_axis(None, axis=1)
        df_new = pd.merge(df1, df, how='outer')

        df_new.to_csv('SP500_new.csv')

stocks(Dater('2018-01-01','2020-05-21') , "SP500")