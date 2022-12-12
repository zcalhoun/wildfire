import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# For processing the tweets
import spacy
from nltk.tokenize import TweetTokenizer


def main(
    input_dir, target_dir, aqi_dir, state, city, county,
):
    """
    This function handles turning the data into a dictionary
    of lemmatized tweets so that we can more easily review
    the data.
    """
    # Read the files from the main directory
    print("Loading tweets...")
    df = load_data(input_dir)

    print("Cleaning dates...")
    # Clean the dates up
    df = clean_dates(df)

    print("Loading AQI files")
    # Load the AQI files
    aqi = load_aqi(aqi_dir, state, county)

    # Get the list of dates:
    dates = df["date"].unique()

    print("Running through dates")
    for d in dates:
        print("Date: ", d)
        # Initialize the dictionary
        tweet_dict = {}
        tweet_dict["AQI"] = aqi[d]

        # Get the tweets for the date
        tweets = df[df["date"] == d]

        # Lemmatize the tweets on this date
        tweet_dict["tweets"] = lemmatize(tweets["text"])

        # Create a json of the lemmatized tweets
        json_data = json.dumps(tweet_dict)

        # Save the json to the target directory
        file_name = str(city) + "_" + str(d) + ".json"
        with open(os.path.join(target_dir, file_name.lower()), "w") as f:
            f.write(json_data)


def lemmatize(tweets):
    """
    This function lemmatizes the tweets.
    """
    lemmatized = []
    tweet_tokenizer = TweetTokenizer()

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stop_words = nlp.Defaults.stop_words
    for tweet in tweets:
        clean_tweet = [
            w for w in tweet_tokenizer.tokenize(tweet.lower()) if w.isalpha()
        ]
        # Skip if the length of the tweet is zero
        if len(clean_tweet) == 0:
            continue

        clean_tweet = [w for w in clean_tweet if not w in stop_words]

        # Continue if length is now zero
        if len(clean_tweet) == 0:
            continue
        doc = nlp(" ".join(clean_tweet))

        # Get the sanitized tweet
        clean_tweet = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 2])
        if len(clean_tweet) == 0:
            continue

        lemmatized.append(clean_tweet)
    return lemmatized


def load_data(input_dir):
    """
    This function loads in the dataframes and returns a pandas
    data frame containing all of the tweets.
    """
    files = []
    for f in os.listdir(input_dir):
        files.append(pd.read_csv(input_dir + f))

    return pd.concat(files)


def load_aqi(aqi_dir, state, county):
    """
    This function uses the city and the county to load the AQI
    for each of the dates.
    """
    files = []
    for f in os.listdir(aqi_dir):
        aqi = pd.read_csv(aqi_dir + f)
        aqi_df = aqi[(aqi["State Name"] == state) & (aqi["county Name"] == county)][
            ["Date", "AQI"]
        ]
        # Ensure that the date is properly formatted as just a simple date object
        aqi_df["Date"] = pd.to_datetime(aqi_df["Date"], format="%Y-%m-%d").apply(
            datetime.date
        )
        # Append all of the dataframes together
        files.append(aqi_df)
    aqi_df = pd.concat(files)
    return aqi_df.set_index("Date").to_dict().get("AQI")


def clean_dates(df):
    """
    This function cleans up the dates in the dataframe.

    This function should discard tweets with dates that
    can't be parsed.
    """
    new_dates = []
    new_tweets = []
    count = 0
    for d, tweet in zip(df["created_at"], df["text"]):
        try:
            cd = datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date()
            new_dates.append(cd)
            new_tweets.append(tweet)
        except:
            count = count + 1

    new_df = pd.DataFrame({"date": new_dates, "text": new_tweets})

    return new_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lemmatize the tweets")
    parser.add_argument(
        "-id", type=str, default="", help="Directory to read the data from"
    )
    parser.add_argument(
        "-td", type=str, default="", help="Directory to save the lemmatized data to",
    )
    parser.add_argument(
        "-ad", type=str, default="", help="Directory to read the aqi data from"
    )
    parser.add_argument("-state", type=str, default="", help="Include the state name")
    parser.add_argument("-city", type=str, default="", help="Include the city name")
    parser.add_argument("-county", type=str, default="", help="Include the county name")
    args = parser.parse_args()

    main(
        args.id, args.td, args.ad, args.state, args.city, args.county,
    )
