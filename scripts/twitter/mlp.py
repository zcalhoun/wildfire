"""
=======================================
A simple MLP in Pytorch for the project
=======================================

desc

1. make_get_aqi: makes a AQI lookup table
2. TweetDataset: a custom Pytorch dataset for tweet data
3. MLP: the MLP model architecture


"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from functools import lru_cache

from nltk.tokenize import TweetTokenizer
import spacy
import nltk
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


"""
========================================================
Builds a AQI lookup dataframe for the TweetDataset class
========================================================
file_dir: location of daily_aqi_by_county_yyyy.csv
location: (state, county)
"""
def make_get_aqi(file_dir, location):
    state, county = location
    aqi = pd.read_csv(file_dir)
    aqi_df = aqi[(aqi['State Name'] == state) & (aqi['county Name'] == county)][['Date', 'AQI']]
    aqi_df['Date'] = pd.to_datetime(aqi_df['Date'], format='%Y-%m-%d').apply(datetime.date)
    return aqi_df.set_index("Date").to_dict().get("AQI")


class TweetDataset(Dataset):
    
    def __init__(self, cv_dict, get_aqi, agg_count=1000, sample_rate=30, random_state=42):

        self.dates = list(cv_dict.keys())
        self.agg_count = agg_count
        self.sample_rate = sample_rate
        self.generator = np.random.default_rng(seed=random_state)
        self.count_vecs = cv_dict
        self.get_aqi = get_aqi
        
    def __len__(self):
        return len(self.dates) * self.sample_rate
    
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        
        date = self.dates[idx % len(self.dates)]
        
        sample = self.generator.choice(self.count_vecs[date].toarray(), self.agg_count, replace=True)
        
        return (
            torch.from_numpy(sample.sum(axis=0)).float().requires_grad_(False),
                            torch.tensor(np.log10(self.get_aqi[date]))
        )

class MLP(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        
        self.fc1 = nn.Linear(vocab, 100)
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)
    
    def loss_function(self, y, y_hat):
        MSE = (y - y_hat).pow(2).mean()
        return MSE