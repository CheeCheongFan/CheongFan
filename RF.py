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
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

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

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 25, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 7]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 55, cv = 3, verbose=2, random_state=41, n_jobs = -1)

x = x_train
y = y_train

rf_random.fit(x,y)



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


base_model = RandomForestRegressor(n_estimators = 125, random_state = 42)
best = base_model.fit(x, y)
base_accuracy = evaluate(base_model, x, y)
best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, x, y)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

y_pred1 = best.predict(x_test)
print("R-Squared: {}".format(r2_score(y_test, y_pred1)))