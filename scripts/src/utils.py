import os
import argparse
import joblib
import re
import logging
import json
from functools import reduce
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

class DataHandler:
    """
    This class handles creating the count vector and vectorizing the
    train and test data.
    """

    def __init__(
        self,
        lemmatized_path,
        storage_path,
        sample_method,
        tweet_agg_num,
        tweet_sample_count,
        min_df,
        max_df,
        train_cities,
        test_cities,
    ):
        self.data_dir = lemmatized_path
        self.storage_path = storage_path
        self.sample_method = sample_method
        self.tweet_agg_num = tweet_agg_num
        self.tweet_sample_count = tweet_sample_count
        self.count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        self.train_cities = train_cities
        self.test_cities = test_cities
        self.files = os.listdir(self.data_dir)

        # If the count vectorizer has already been created, load it.
        if os.path.exists(self.storage_path):
            if os.path.exists(os.path.join(self.storage_path, "train_data.joblib")):
                self.train_data = joblib.load(
                    os.path.join(self.storage_path, "train_data.joblib")
                )
            else:
                self.train_data = None

            if os.path.exists(os.path.join(self.storage_path, "test_data.joblib")):
                self.test_data = joblib.load(
                    os.path.join(self.storage_path, "test_data.joblib")
                )
            else:
                self.test_data = None

    def create_train_dataset(
        self,
    ):

        if self.train_data is not None:
            return self.train_data

        # Else, create the training data based on the count vectorizer

        cities = "|".join(self.train_cities)
        train_files = list(filter(lambda x: re.search(cities, x), self.files))

        # Load all of the training data
        train_data_json, train_tweets = load_data(self.data_dir, train_files)

        # Concatenate the tweets into a single array
        # This is needed to create the count vectorizer
        train_tweets = reduce(lambda x, y: x + y, train_tweets)

        self.count_vectorizer.fit(train_tweets)
        # Save the count vectorizer for later use
        joblib.dump(
            self.count_vectorizer,
            os.path.join(self.storage_path, "count_vectorizer.joblib"),
        )

        # Turn the training data into a joblib file
        self._count_vectorize_and_save(train_data_json, "train_data.joblib")

    def create_test_dataset(
        self,
    ):
        if self.test_data is not None:
            return self.test_data

        cities = "|".join(self.test_cities)
        test_files = list(filter(lambda x: re.search(cities, x), self.files))

        # Load all of the test data
        test_data_json, test_tweets = load_data(self.data_dir, test_files)

        self._count_vectorize_and_save(test_data_json, "test_data.joblib")

    def _count_vectorize_and_save(
        self,
        data,
        filename,
    ):
        """
        This function takes in a data set and vectorizes it.
        """
        output_file = []
        for day_city in data:
            tweets = self.count_vectorizer.transform(day_city["tweets"]).toarray()

            if sample_method == "by_file":
                sample_rate = self.tweet_sample_count
            else:
                sample_rate = (
                    int(tweets.shape[0]) / self.tweet_agg_num * self.tweet_sample_count
                )

            aqi = day_city["AQI"]
            generator = np.random.default_rng(seed=42)

            # Generate all of the samples for this day/city combo
            for i in range(0, sample_rate):
                sample = generator.choice(tweets, self.tweet_agg_num, replace=False)

                output_file.append({"api": aqi, "tweets": sample.sum(axis=0)})

        joblib.dump(output_file, os.path.join(self.storage_path, filename))


def load_data(data_dir, files):
    """
    Loads the data from the files in the directory.
    """
    json_data = []
    tweets = []
    for file in files:
        with open(os.path.join(data_dir, file)) as f:
            data = json.load(f)
            json_data.append(data)
            tweets.append(data["tweets"])
    return json_data, tweets
