"""
=======================================
A simple MLP in Pytorch for the project
=======================================

desc

1. make_get_aqi(): makes a AQI lookup table
2. make_get_vect(): makes a vectorized lookup table
3. TweetDataset: a custom Pytorch dataset for tweet data
4. train_model(): trains a custom Pytorch NN
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache

import torch
from torch.utils.data import Dataset

# Builds a raw dataframe of all tweets from selected cities
# ---------------------------------------------------------
# base_dir: location of twitter data 
# cities: names of cities
def make_tweet_df(base_dir, cities):
    # Reads files in a directory as csv and returns dataframe
    def concat_data(base_dir):
        files = os.listdir(base_dir)
        dfs = []
        for f in files:
            if ".csv" in f:
                dfs.append(pd.read_csv(base_dir + f))
        return pd.concat(dfs)
    
    dfs = []

    for city in cities:
        df = concat_data("{}/{}/".format(base_dir, city))
        df['city'] = city
        dfs.append(df)
        
    return pd.concat(dfs)

# Builds a vectorized lookup table for the TweetDataset class
# -----------------------------------------------------------
# df: dataframe with "date", "city", and "text" (cleaned) columns
# vectorizer: Any sklearn vectorizer already fit (e.g. CountVectorizer)
def make_get_vect(df, vectorizer):
    x_vect = vectorizer.transform(df.text)
    dates, cities = df.date.unique(), df.city.unique()

    # Calling unique converts dates from timestamp to numpy.datetime64
    dates = [pd.Timestamp(date) for date in dates]

    x_dict = {}
    for date in dates:
        for city in cities:
            x_dict[(date,city)] = x_vect[(df.date == date) & (df.city == city)]
    return x_dict


# Builds a dataframe of all AQI readings for given locations and years
# base dir: (e.g. ../data)
# years: list of target years
# locations: (state, country, city)
def make_aqi_df(base_dir, years, locations):
    # Load AQI data for each year
    aqi_raw = []
    for year in years:
        aqi_raw.append(pd.read_csv('{}/daily_aqi_by_county_{}.csv'.format(base_dir, year)))
    aqi_raw = pd.concat(aqi_raw)

    # Subset AQI to just wanted counties
    aqi_df = []
    for county, state, city in locations:
        aqi_temp = aqi_raw[(aqi_raw['State Name']==state) & (aqi_raw['county Name']==county)] 
        # Need to create a city key in AQI df for merging later
        aqi_temp["city"] = city
        aqi_df.append(aqi_temp)

    # Subset concat and extract dates
    aqi_df = pd.concat(aqi_df)
    aqi_df['date'] = pd.to_datetime(aqi_df['Date'])

    return aqi_df


# Builds a AQI lookup dataframe for the TweetDataset class
# --------------------------------------------------------
# aqi_df: dataframe with "date", "city", and "AQI" columns
def make_get_aqi(aqi_df):
    return aqi_df.set_index(["date", "city"]).to_dict().get("AQI")


# Implementation of Pytorch Dataset for Tweets and AQI
# ----------------------------------------------------
# get_vect: map from a date and city to list of tweet vectors
# get_aqi: map from a data and city to the AQI
# agg_count: total tweets sampled per request
# sample_rate: number of samples per day
# random_state: ...
class TweetDataset(Dataset):
    
    def __init__(self, get_vect, get_aqi, agg_count=1000, sample_rate=30, random_state=42):

        self.dates, self.cities = zip(*get_vect.keys())
        self.agg_count = agg_count
        self.sample_rate = sample_rate
        self.generator = np.random.default_rng(seed=random_state)
        self.get_vect = get_vect
        self.get_aqi = get_aqi
        
    def __len__(self):
        return len(self.dates) * self.sample_rate
    
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        
        idx %= len(self.dates)
        date = self.dates[idx]
        city = self.cities[idx]
        
        sample = self.generator.choice(self.get_vect[(date, city)].toarray(), self.agg_count, replace=True)
        
        return (
            torch.from_numpy(sample.sum(axis=0)).float().requires_grad_(False),
                            torch.tensor(np.log10(self.get_aqi[(date, city)]))
        )


# Trains the given model on the given data
# ----------------------------------------
# model: Pytorch neural network
# train_loader: Pytorch train data loader
# test_loader: Pytorch test data loader
# optimizer: Pytorch optimizer
# epochs: number of training epochs
# print_rate: controls the amount of output printed
def train_model(model, train_loader, test_loader, optimizer, epochs, print_rate):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")

    loss_results = {
        "train": [],
        "val": []
    }

    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_test_loss = 0
        # Run method on training
        model.train()
        for batch_idx, (data, y) in enumerate(train_loader):
            # Add training data to GPU
            data = data.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(data)
            
            y_hat = y_hat.to(device)
            
            loss = model.loss_function(y, y_hat)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % print_rate == 0:
                print(
                    "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        print(
            "===> Epoch: {} Average Loss: {:.4f}".format(
                epoch, epoch_train_loss / batch_idx
            )
        )
        loss_results['train'].append(epoch_train_loss/batch_idx)

        # loss['train'].append(epoch_train_loss)

        # Capture testing performance.
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, y) in enumerate(test_loader):
                # Add to GPU
                data = data.to(device)
                y = y.to(device)
                y_hat = model(data)
                loss = model.loss_function(y, y_hat)
                epoch_test_loss += loss.item()

        epoch_test_loss /= batch_idx
        if epoch_test_loss < 0.05:
            break
        # Append results to the json
        loss_results["val"].append(epoch_test_loss)

        print("===> Test set loss: {:.4f}".format(epoch_test_loss))
        
        return loss_results
