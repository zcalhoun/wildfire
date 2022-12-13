import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp

# For processing the tweets
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

    date_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    print(f"The slurm id is {date_id}")

    if len(dates) <= date_id:
        print("The date id is too large")
        return
    d = dates[date_id]

    tweets = df[df["date"] == d]

    day_tweet = DayTweet(d, tweets['text'], aqi[d], city, target_dir)

    day_tweet.lemmatize_and_save()
    print(f"Created file for {d}")

def create_file(d):
    d.lemmatize_and_save()
    return d.date

class DayTweet():
    def __init__(self, date, text, aqi, city, target_dir):
        self.date = date
        self.text = text
        self.aqi = aqi
        self.city = city
        self.target_dir = target_dir

    def lemmatize_and_save(self):
        # Create a place to hold the information
        tweet_dict = {}
        tweet_dict["AQI"] = self.aqi

        # Apply the lemmatize function
        tweet_dict["tweets"] = self.lemmatize()

        # Create a json of the lemmatized tweets
        json_data = json.dumps(tweet_dict)

        # Save the json to the target directory
        file_name = str(self.city) + "_" + str(self.date) + ".json"
        with open(os.path.join(self.target_dir, file_name.lower()), "w") as f:
            f.write(json_data)

    def lemmatize(self,):
        """
        This function lemmatizes the tweets.
        """
        lemmatized = []
        tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True, match_phone_numbers=False)

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        for tweet in self.text:
            # Tokenize the tweets
            clean = tweet_tokenizer.tokenize(tweet)

            # Remove non-unicode characters
            clean = [w.encode('ascii', errors='ignore').decode() for w in clean]
            
            # Skip if the length of the tweet is zero
            if len(clean) == 0:
                continue

            # Remove punctuation/emojis and 2 letter words
            clean = [w for w in clean if len(w) > 2]

            # Skip if the length of the tweet is zero
            if len(clean) == 0:
                continue

            # Remove all of the stop words
            clean = [w for w in clean if w not in stop_words]

            # Skip if the length of the tweet is zero
            if len(clean) == 0:
                continue

            # Lemmatize the tweet
            clean = [lemmatizer.lemmatize(w) for w in clean]

            # Get the sanitized tweet
            clean = " ".join(clean)

            lemmatized.append(clean)
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
