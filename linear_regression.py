import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from os import listdir
from os.path import join
from vocab import Vocab

## def export_csv(dic):
##     df = pd.DataFrame.from_dict(dic)
##     df.to_csv('sentiments/The_Guardian.csv')
##     print("Sentiment csv successfully created.")

def calculate_avg(sentiment_lst):
    dflst = [pd.read_csv(csv, sep=',', index_col=0) for csv in sentiment_lst]
    
    res = dflst[0].iloc[:,:1]
    res.loc[:,'Sentiment'] = 0
    res.loc[:,'Textblob_polarity'] = 0
    res.loc[:,'Textblob_Emotional'] = 0

    for df in dflst:
        for i in range(len(res['Date'])):
            if list(res['Date'])[i] in list(df['Date']) and df.loc[i, 'Count'] != 0:
                res.loc[i,'Sentiment'] += df.loc[i, 'Sentiment']/df.loc[i, 'Count']
                res.loc[i,'Textblob_polarity'] += df.loc[i, 'Textblob_polarity']/df.loc[i, 'Count']
                res.loc[i,'Textblob_Emotional'] += df.loc[i, 'Textblob_Emotional']/df.loc[i, 'Count']
            else:
                res.loc[i,'Sentiment'] += 0
                res.loc[i,'Textblob_polarity'] += 0
                res.loc[i,'Textblob_Emotional'] += 0
    return res

def importmerge_stock(csv, df):
    fields = ['chg', 'Date']
    df_stock = pd.read_csv(csv, sep=',', usecols=fields)

    df3 = pd.merge(df, df_stock, how='outer')
    res = df3[df3['chg'].notna()]

    res.to_csv("merged.csv")

    return res

def plot(df):
    if 'Sentiment' in df and 'chg' in df and 'Textblob_polarity' in df and 'Textblob_Emotional' in df:
        # generate summary table
        print("Summary for chg against Sentiment...")
        mod1 = sm.ols("chg ~ Sentiment", data=df).fit()
        print(mod1.summary())
        print("\n\n\n")
        
        print("Summary for chg against Textblob polarity...")
        mod2 = sm.ols("chg ~ Textblob_polarity", data=df).fit()
        print(mod2.summary())
        print("\n\n\n")

        print("Summary for chg against Textblob Emotional...")
        mod3 = sm.ols("chg ~ Textblob_Emotional", data=df).fit()
        print(mod3.summary())

        print("Summary for chg against all 3...")
        mod = sm.ols("chg ~ Sentiment + Textblob_polarity + Textblob_Emotional", data=df).fit()
        print(mod.summary())
        # plot regression line
        fig, axs = plt.subplots(ncols=3)
        plot1 = sns.regplot(x='Sentiment', y='chg', data=df, ax=axs[0], ci=None)
        plot2 = sns.regplot(x='Textblob_polarity', y='chg', data=df, ax=axs[1], ci=None)
        plot3 = sns.regplot(x='Textblob_Emotional',y='chg', data=df, ax=axs[2], ci=None)
        # changing limit of axes
        plot1.set(xlim=(min(df['Sentiment'])-2, max(df['Sentiment'])+2))
        plot2.set(xlim=(min(df['Textblob_polarity'])-0.02, max(df['Textblob_polarity'])+0.02))
        plot3.set(xlim=(min(df['Textblob_Emotional'])-0.02, max(df['Textblob_Emotional'])+0.02))
##        plot = sns.regplot(x='Sentiment', y='chg', data=df, ci=None)
##        plot.set(xlim=(min(df['Sentiment'])-10, max(df['Sentiment'])+10))
        plt.show()
    else:
        print("Invalid dataframe as input")
        return

if __name__ == "__main__":
    sentiment_csvs = [join("sentiments", csv) for csv in listdir("sentiments")]
    avg_sentiment = calculate_avg(sentiment_csvs)
    print(avg_sentiment)
    res = importmerge_stock("sp500.csv", avg_sentiment)
    print(res)
##    plot(res)
