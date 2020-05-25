from os import listdir
from os.path import join
import joblib
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
from sklearn import preprocessing
from sklearn import utils
from Automator.linear_regression import calculate_avg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import stats


source = 'New_York_Times'
sentiments = pd.DataFrame(pd.read_csv('sentiments/{}.csv'.format(source),sep='\s*,\s*'))
change = pd.DataFrame(pd.read_csv('SP500.csv'))
merged = pd.DataFrame(pd.merge(sentiments, change, on="Date"))
new = merged[['Date','Textblob_polarity', 'Textblob_Emotional', 'Sentiment','senti','tb_pol','chg']].copy()
sentiment_csvs = [join("sentiments", csv) for csv in listdir("sentiments")]
avg = calculate_avg(sentiment_csvs)

avg_renamed = avg.rename(columns={'Textblob_polarity':'Textblob_polarity_avg',
                                   'Textblob_Emotional':'Textblob_Emotional_avg',
                                    'Sentiment':'Sentiment_avg'})
mergedavg = pd.merge(new, avg_renamed, on = "Date")

mergedavg.to_csv('merged_{}.csv'.format(source))

df = pd.read_csv('merged_{}.csv'.format(source))
df['Textblob_polarity_avg_dayb4'] = df['Textblob_polarity_avg'].shift(-1)
df['Textblob_Emotional_avg_dayb4'] = df['Textblob_Emotional_avg'].shift(-1)
df['Sentiment_avg_dayb4'] = df['Sentiment_avg'].shift(-1)
df.dropna(subset = ["Textblob_polarity_avg_dayb4","Textblob_Emotional_avg_dayb4","Sentiment_avg_dayb4"], inplace = True)
num_train = df.select_dtypes(include=["number"])
cat_train = df.select_dtypes(exclude=["number"])
idx = np.all(stats.zscore(num_train) < 3, axis=1)

cleaned = pd.concat([num_train.loc[idx], cat_train.loc[idx]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(cleaned['Textblob_polarity_avg_dayb4'],cleaned['chg'], test_size = 0.2)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

rf = RandomForestRegressor(random_state = 42)
rf_model = rf.fit(x_train, y_train)
y_pred1 = rf.predict(x_test)


X_grid = np.arange(min(x_train), max(x_train), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x_train, y_train, color = 'red')
plt.plot(X_grid, rf_model.predict(X_grid), color = 'blue')
plt.title('Check It (Random Forest Regression Model)')
plt.xlabel('polarity')
plt.ylabel('chg')
plt.show()
#print(r2_score(y_test, y_pred1))
x = cleaned[['Textblob_polarity_avg_dayb4','Textblob_Emotional_avg_dayb4','Sentiment_avg_dayb4']].values
y = cleaned['chg'].values
x = np.array(x)
y = np.array(y)

x_train_lin, x_test_lin, y_train_lin, y_test_lin = train_test_split(x,y,test_size=0.2, random_state=0)

clf = LinearRegression(normalize=True)
clf.fit(x_train_lin, y_train_lin)
y_pred = clf.predict(x_test_lin)
print('Accuracy of LR',mean_squared_error(y_pred,y_test))
print("Linear Regression: {}".format(r2_score(y_test,y_pred)))
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))


plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()