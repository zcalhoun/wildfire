import numpy as np
import re
import pandas as pd
from datetime import datetime
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

# Set up global variables
num_topics = 10

# Get the data
def load_data():
    """
    This function takes care of loading data, returning a dataframe
    with the tweets that we want to look at.

    If you are going to change anything on this file, change it here.
    """
    try:
        months = ['05','06','07','08','09','10','11','12']
        all_df = []
        for m in months:
            all_df.append(pd.read_csv('../data/san_francisco/2018-'+m+'.csv'))
       # Concatenate all dataframes into one DF. 
        all_df = pd.concat(all_df)
        return all_df
    except:
        print("Files not found")

def format_data(data):
    """
    Add special formatting instructions here
    The output of this function should be the pseudo docs that will be read
    by the LDA function.
    """
    data = data.dropna()
    # not_date = [d for d in all_df['created_at'] if type(d) ==float]
    data['date'] = [datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in data['created_at']]
    pseudo_docs = pd.pivot_table(data,values="text",index="date",aggfunc=" ".join)
    # Only output the docs
    docs = pseudo_docs['text']
    return docs

def normalize_words(docs):
    """
    This function should take in the array of documents, and for each
    document, we should:
        1. Make the words lowercase
        2. Tokenize the words
        3. Remove stopwords and non alphabetic words

    The output of the function should the normalized docs.
    """
    cleaned_tweets = []
    lemmatizer = WordNetLemmatizer()
    tt = TweetTokenizer()
    stopword_list = nltk.corpus.stopwords.words("english")

    for d in docs:
        t = d.lower()
        t = tt.tokenize(t)
        t = [w for w in t if w.isalpha()]
        t = [lemmatizer.lemmatize(w) for w in t]
        t = [w for w in t if w not in stopword_list]
        cleaned_tweets.append(t)
    # This is so we can pass this value into the count vectorizer
    ct = [" ".join(t) for t in cleaned_tweets]
    return ct

def get_count_vec(docs,max_df):
    """
    This function takes in the docs and returns the count vector
    and the term_frequency for the docs provided.
    """
    cv = CountVectorizer(stop_words='english', max_df=max_df, min_df=5)
    term_freq = cv.fit_transform(docs)
    feature_names = cv.get_feature_names_out()
    return cv, term_freq

def main():
    print("In main")
    # Load the data
    data = load_data()
    # Format the data - this should 
    # organize the words in whatever form we want
    # for running through LDA.
    pseudo_docs = format_data(data)
    # Clean up the words
    pseudo_docs = normalize_words(pseudo_docs)
    # Create count vectors and the term frequency
    count_vec,tf = get_count_vec(pseudo_docs,.95)
    # Run LDA on this data.
    lda = LatentDirichletAllocation(max_iter=1000, n_components=num_topics, random_state=42)
    topic_assignment = lda.fit_transform(tf)
    # Save the model for later
    dump(lda, 'lda.joblib')
    dump(count_vec, 'cv.joblib')
    dump(tf, 'tf.joblib')
    dump(topic_assignment, 'topic_assignment.joblib')

if __name__ == "__main__":
    # Load the data
    main()
