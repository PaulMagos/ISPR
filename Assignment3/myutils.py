import pandas as pd
import os

def loadOriginal(realPerc = 1.0, fakePerc = 1.0):
    # Load data
    fakeNews = pd.read_csv('./archive/Fake.csv')
    # Lowercase text and title
    fakeNews.text = fakeNews.text.str.lower()
    fakeNews.title = fakeNews.title.str.lower()
    realNews = pd.read_csv('./archive/True.csv')
    # Lowercase text and title
    realNews.text = realNews.text.str.lower()
    realNews.title = realNews.title.str.lower()
    # Add label column to both dataframes
    realNews['label'] = 0
    fakeNews['label'] = 1
    return pd.concat([fakeNews, realNews]).sample(frac=1).reset_index(drop=True)

def initDF():
    # Load data, if not present create it and save it
    if os.path.exists("./archive/allNews.csv"):
        print("Loading data from processed file")
        allNews = pd.read_csv('./archive/allNews.csv')
    else:
        print("Loading data from original files")
        allNews = loadOriginal()
        allNews.to_csv('./archive/allNews.csv', index=False)
    allNews.apply(lambda x: x.astype(str).str.lower())
    return allNews

def unbalance(df, labels, kind = 0):
    if os.path.exists(f"./archive/allNewsTrain{kind}.csv"):
        df = pd.read_csv(f"./archive/allNewsTrain{kind}.csv")
        labels = df.pop('label')
    else:
        toDrop = df[labels == kind].sample(frac=.5).index
        df.drop(toDrop, inplace=True)
        labels.drop(toDrop, inplace=True)
        df1 = df
        df1['label'] = labels
        df1.to_csv(f'./archive/allNewsTrain{kind}.csv', index=False)
    
    return (df, labels)