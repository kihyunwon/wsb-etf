from datetime import datetime
import math
import re
import threading

import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
import pandas as pd
import praw


# init sentiment model
sia = SIA()

# fill your credentials
client_id = ''
client_secret=''
user_agent=''
keywords = {'dd', 'moon', 'squeeze', 'yolo', 'calls', 'undervalued', 'strong', 'gains'}


def download_library():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    try:
        nltk.data.find('stopwords')
    except:  
        nltk.download('stopwords')


def load_tickers():
    nasdaq = pd.read_csv('tickers/nasdaq.csv')
    nyse = pd.read_csv('tickers/nyse.csv')
    amex = pd.read_csv('tickers/amex.csv')
    return np.union1d(np.union1d(nasdaq.Symbol.values, nyse.Symbol.values), amex.Symbol.values)


def wsb_client():
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
    return reddit.subreddit('wallstreetbets')


def ticker_from_text(tickers, text):
    cap = text.split()
    filtered = list(filter(lambda x: x in tickers, cap))
    filtered.sort(key=lambda x: len(x))
    # return longest capitalized word
    return filtered[-1] if len(filtered) > 0 else ''


def search_ticker(tickers, title, text):
    ticker = ticker_from_text(tickers, title)
    if not ticker:
        return ticker_from_text(tickers, text)


def analyze_post(comments):
    try:
        body = []
        for comment in comments:
            body.append(comment.body)
        results = []
        for line in body:
            scores = sia.polarity_scores(line)
            scores['raw_text'] = line
            tokens = line.lower().split()
            # extra boost for every keyword found
            for token in tokens:
                if token in keywords:
                    scores['compound'] += 0.1
            results.append(scores)
        df = pd.DataFrame.from_records(results)
        df['sentiment'] = 0
        df.loc[df['compound'] > 0.1, 'sentiment'] = 1
        df.loc[df['compound'] < -0.1, 'sentiment'] = -1
        return df.sentiment.mean()
    except Exception as e:
        print(e)
        return 0


def get_date(date):
    return datetime.fromtimestamp(date)


def crawl(tickers):
    wsb = wsb_client()

    results = {}
    posts = wsb.new(limit=100)
    for post in posts:
        d = {}
        ticker = search_ticker(tickers, post.title, post.selftext)
        # ticker is not found
        if not ticker:
            continue
        sentiment = analyze_post(post.comments)
        # skip neutral post
        if sentiment == 0:
            continue
        d['ticker'] = ticker
        d['avg_comment_sentiment'] = sentiment
        d['num_comments'] = post.num_comments
        d['score'] = post.score
        d['upvote_ratio'] = post.upvote_ratio
        d['created_date_utc'] = get_date(post.created_utc)
        d['author'] = post.author
        # promote post with higher sentiment if the ticker already exists
        if ticker in results:
            old_d = results[ticker]
            if abs(old_d['avg_comment_sentiment']) > abs(d['avg_comment_sentiment']):
                d = old_d
        results[ticker] = d
    
    df = pd.DataFrame(results)
    df = df.sort_values(by='avg_comment_sentiment', axis=1, ascending=False)
    print(df.iloc[:, :10])
    # save for the record
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    df.to_csv('{}.csv'.format(now), index=False)
    # crawl wsb hourly
    threading.Timer(60, lambda: crawl(tickers)).start()


if __name__ == '__main__':
    download_library()
    tickers = load_tickers()
    crawl(tickers)
