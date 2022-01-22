import matplotlib.pyplot as plt
import cartopy # Get rid of cartopy if you don't care about creating beautiful maps!
import cartopy.feature as cfeature # Same here, too.
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from requests.auth import HTTPBasicAuth
import requests
import json

def plot_tweets(tweets_arr, img_extent=(-120, -75,21, 50)):

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_axes([0, 0, 1, 1], projection=cartopy.crs.LambertConformal(central_longitude=-98.0))
    ax.set_extent(img_extent, cartopy.crs.Geodetic())
    ax.coastlines()

    # Add state boundaries
    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

    ax.add_feature(states_provinces, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS)
    for i, t in enumerate(tweets_arr):
      ax.scatter(tweets_arr[i]['lon'],tweets_arr[i]['lat'], transform=cartopy.crs.PlateCarree())
    
    # plt.show()

def graph_tweet_count(dfs, startDate, endDate, labels=[]):
    fig, ax = plt.subplots(figsize=(16, 9))
    for i, df in enumerate(dfs):
        df = df.sort_values('start', ascending=True)
        df['day'] = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%fZ') for d in df['start']]
        if(len(labels)>0):
            print(labels[i])
            ax.plot(df['day'], df['tweet_count'], label= labels[i])
        else:
            ax.plot(df['day'], df['tweet_count'])
        ax.set(xlabel="Date (Month/Year)",
              ylabel="# of Tweets",
              title="Mentions",
                xlim=[startDate, endDate])

        date_form = DateFormatter("%m/%y")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    if(len(labels)>0):
        plt.legend()
    plt.show()

def generate_bearer_token(api_key, api_key_secret):
    params = {
        "grant_type":"client_credentials"
    }
    r = requests.post("https://api.twitter.com/oauth2/token", params=params, auth=HTTPBasicAuth(api_key,api_key_secret))
    
    if(r.status_code==200):
        print("Token generated")
    else:
        print(r.text)
        return
       
    data = {
        "bearer_token": r.json()['access_token']
    }
    
    with open('.twitter_creds.json', 'w') as outfile:
        json.dump(data, outfile)
        
    return True
