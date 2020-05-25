import nltk
import string
import re
import os
import datetime
import requests
import json
import textblob
import csv
import pandas as pd
from afinn import Afinn
from textblob import TextBlob
from nltk.corpus import stopwords
from datetime import datetime

afinn = Afinn()

stopwords = stopwords.words('english')
wn = nltk.WordNetLemmatizer()


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text


def textblob(data):
    cleaned = clean_text(data)
    testimonial = TextBlob(' '.join(cleaned))
    return testimonial.sentiment.polarity, testimonial.sentiment.subjectivity


def afinn_score(data):
    score = int(afinn.score(data))
    return score


def plusminus(value):
    test = float(value)
    if test > 0:
        return 1
    elif test < 0:
        return -1
    else:
        return 0


def calculate(folder, path, score_dic, files_dic, results_dic, index, cal_bef=False):
    temp = 0
    count = len(os.listdir(path))
    for file in os.listdir(path):
        filename = os.path.join(path, file)
        if os.path.isfile(filename) \
            and filename.endswith(".txt") \
            and not filename in files_dic:
            with open(filename, "r", encoding="utf-8") as file:
                files_dic[filename] = file.read().replace('\n', '')
                data = files_dic[filename]
                tb_polarity, tb_emo = textblob(data)

                score_dic['afinn'].append(afinn_score(data))
                score_dic['tb_polarity'].append(tb_polarity)
                score_dic['tb_emo'].append(tb_emo)

    total_tb_polarity = sum(score_dic['tb_polarity'])
    total_emo = sum(score_dic['tb_emo'])
    total_afinn = sum(score_dic['afinn'])
    print("Score for the date of {}: {} for afinn, {} for TextBlob with {} articles".format(folder, total_afinn,
                                                                                                    total_tb_polarity, count))
    print("Emotional Score: {}".format(total_emo))
    if cal_bef:
        results_dic['Textblob_polarity'][index] = total_tb_polarity
        results_dic['Textblob_Emotional'][index] = total_emo
        results_dic['Sentiment'][index] = total_afinn
        results_dic['Count'][index] = count
        results_dic['tb_pol'][index] = plusminus(total_tb_polarity)
        results_dic['senti'][index] = plusminus(total_afinn)
    else:

        results_dic['Textblob_polarity'].append(total_tb_polarity)
        results_dic['Textblob_Emotional'].append(total_emo)
        results_dic['Sentiment'].append(total_afinn)
        results_dic['Count'].append(count)
        results_dic['tb_pol'].append(plusminus(total_tb_polarity))
        results_dic['senti'].append(plusminus(total_afinn))
    return True


def Vocab(fold):
    update = False
    x = os.path.join(os.getcwd(), '{}'.format(fold))
    print("Reading directory: {}".format(x))
    files = {}
    results = {
        'Date': [],
        'Textblob_polarity': [],
        'Textblob_Emotional': [],
        'Sentiment': [],
        'Count': [],
        'tb_pol': [],
        'senti': []}

    try:
        path_to_csv = os.path.join('sentiments', fold + '.csv')
        with open(os.path.join('sentiments', fold + '.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for i in reader:
                try:
                    results['Date'].append(datetime.strptime(i[1], '%d/%m/%Y').strftime('%Y-%m-%d'))
                    print("Changed date format")
                except:
                    results['Date'].append(i[1])

                results['Textblob_polarity'].append(i[2])
                results['Textblob_Emotional'].append(i[3])
                results['Sentiment'].append(i[4])
                results['Count'].append(i[5])
    except ValueError:
        pass
    except OSError:
        print("Sentiment .csv not found, generating new .csv...")

    for folder in os.listdir(x):
        # checks if all articles for the day are calculated
        path = os.path.join(x, folder)
        count = 0

        scores = {
            'afinn': [],
            'tb_polarity': [],
            'tb_emo': []}

        if os.path.isdir(path):
            if folder in results['Date']:
                count = len(os.listdir(path))
                index = results['Date'].index(folder)
                if count == int(results['Count'][index]):
                    print("Values for the date of {} already calculated, skipping...".format(folder))
                    continue
                else:
                    update = calculate(folder, path, scores, files, results, index, cal_bef=True)
            else:
                results['Date'].append(folder)
                update = calculate(folder, path, scores, files, results, None)

    # export sentiments csv
    if update:
        os.makedirs('sentiments', exist_ok=True)
        print([len(elements) for elements in results])
        df = pd.DataFrame.from_dict(results)
        df.to_csv('sentiments/{}.csv'.format(fold))
        print("Sentiment csv successfully created.")
    else:
        print("No need to update")


if __name__ == "__main__":
    folders = ["New_York_Times"]
    for folder in folders:
        Vocab(folder)


