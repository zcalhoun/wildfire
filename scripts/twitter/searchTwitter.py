# This piece of code was made to simplify future analyses. We can
# take this piece of code out and put into a script to be used
# in future Jupyter notebooks. This will drastically simplify 
# the amount of work required to make a request.

import time # required to ensure we don't request from Twitter too frequently (limit 1 call /sec)
import json
from datetime import datetime
from collections import defaultdict, namedtuple
import requests
import pandas as pd

class TwitterDataFrame(pd.DataFrame):
    """
    This is a subclass of the pandas dataframe that allows
    me to more quickly look at data in a specific way.
    --
    I've added several functions to make working with tweets
    easier. Functionality includes:
    --
    * zoom_in(lat, lon) -- set a pair of lat/lon tuples, and this
        function will return a dataframe that filters out features
    """
    
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    def count_by_day(self):
        """
        This function will go ahead and return a pivot table on this data
        so that you can quickly look at tweet counts by day for this dataframe.
        """
        if("date" not in self.columns):
            self['date'] = [d.date() for d in self['created_at']]
        return pd.pivot_table(self, values='id', index='date',aggfunc='count')
        
    def get_coords(self):
        lat = (min(self['lat']), max(self['lat']))
        lon = (min(self['lon']), max(self['lon']))
        return {
            "lat":lat,
            "lon":lon
        }

    def group_by_day(self):
        """
        This function agreggates tweets by date for further analysis.

        TODO: Test this out.
        """
        if("date" not in self.columns):
            self['date'] = [d.date() for d in self['created_at']]
        pseudo_docs = pd.pivot_table(self,values="text",index="date",aggfunc=" ".join)
        return pseudo_docs
        
    def zoom_in(self, lat, lon):
        """
        Pass in a lat tuple and a lon tuple, 
        and this function will return a new TwitterDataFrame
        with only the tweets that fill within those coordinates.
        """
        lat1 = min(lat); lat2 = max(lat)
        lon1 = min(lon); lon2 = max(lon)
        
        mask1 = (self['lon'] > lon1) & (self['lon'] < lon2)
        mask2 = (self['lat'] > lat1) & (self['lat'] < lat2)
        
        return TwitterDataFrame(self[mask1 & mask2])

class TwitterSearchTerm():
    def __init__(self, queryString, startTime, endTime):
        self.queryString = queryString
        self.startTime = startTime
        self.endTime = endTime
        self.countDataFrame = None
        self.tweet_count = None 
        
    def create_header(self):
        """
        This function reads from the json file that stores the bearer
        token for authenticating requests
        """
        with open('./.twitter_creds.json') as f:
            data = json.load(f)

        bearer = data["bearer_token"]
        headers = {
            'Authorization': 'Bearer '+bearer
        }
        
        return headers
    
    def get_term_count(self):
        payload = {
            "query": self.queryString,
            "start_time":self.startTime,
            "end_time":self.endTime,
            "granularity":"day"
        }

        ## Create get request
        try:
            response = requests.get("https://api.twitter.com/2/tweets/counts/all", params=payload, headers=self.create_header())
        except Exception as e:
            print("=========================")
            print("An error occurred with the request.")
            print(response['text'])
            print(e)
            print("=========================")
            raise
        else: # Check that the status code is 200
            if (response.status_code != 200):
                print("=========================")
                print("An error occurred.")
                for i, error in enumerate(response.json()['errors']):
                    print("Error ", i, ":")
                    print(error['message'])
                    print("=========================")
                    raise
        
        responses = [response.json()["data"]]
        total_tweets = response.json()["meta"]["total_tweet_count"]

        # if there are more results, get these here.
        while("next_token" in response.json()["meta"]):
            payload["next_token"] = response.json()["meta"]["next_token"]
            response = requests.get("https://api.twitter.com/2/tweets/counts/all", params=payload, headers=self.create_header())
            total_tweets = total_tweets + response.json()["meta"]["total_tweet_count"]
            responses.append(response.json()["data"])

        # Return complete dataframe with all tweets
        frames = []
        for r in responses:
            r_str = json.dumps(r)
            frames.append(pd.read_json(r_str))

        # Print out total tweets counted for query
        print("===========================================")    
        print("Total tweets", total_tweets, "for query", self.queryString)
        print("===========================================")

        df = pd.concat(frames)
        self.countDataFrame = df
        self.tweet_count = total_tweets
        return df
    
    def get_count_data_frame(self):
        return self.countDataFrame
    
    def get_tweet_count(self):
        return self.tweet_count
    
    def process_coordinates(self,tweet_set):
#     """
#     This function is required to get estimated
#     coordinates of the tweet locations.
    
#     """
        coords = namedtuple('coords', 'lon lat')
        coordinate_map = defaultdict(coords)

        if('includes' in tweet_set):
            for location in tweet_set['includes']['places']:
                coordinate_map[str(location['id'])] = coords(*location['geo']['bbox'][0:2])

        lat = []
        lon = []
        for tweet in tweet_set["data"]:
            if('geo' in tweet):
                if('coordinates' in tweet['geo']):
                    lat.append(tweet['geo']['coordinates']['coordinates'][1])
                    lon.append(tweet['geo']['coordinates']['coordinates'][0])
                else:
                    place_id = str(tweet['geo']['place_id'])
                    if(place_id in coordinate_map):
                        lat.append(coordinate_map[place_id].lat)
                        lon.append(coordinate_map[place_id].lon)
                    else:
                        lat.append(-9999)
                        lon.append(-9999)
            else:
                lat.append(-9999)
                lon.append(-9999)
                

        return (lat, lon)
    
    def get_tweets(self, max_results = 500):
        """
        This function retrieves the tweets from Twitter's API and 
        returns the tweets as a DataFrame.
        """
        payload = {
            "query": self.queryString,
            "start_time":self.startTime,
            "end_time":self.endTime,
            "max_results": max_results,
            "tweet.fields":"geo,created_at",
            "expansions":"geo.place_id,author_id",
            "place.fields":"geo,contained_within,country,full_name,name"
        }

        request_num, total_requests = (1, None) # Counter for tracking requests
        if(self.tweet_count is None):
            print("Making request 1 of N")
        else:
            total_requests = int(self.tweet_count/max_results)+1
            print("Making request 1 of %d" % total_requests)
        response = requests.get("https://api.twitter.com/2/tweets/search/all", params=payload, headers=self.create_header())
            
        if (response.status_code != 200):
            print("==============================")
            print("The status is not 200. Response: ")
            print(response)
            print("Payload:")
            print(payload)
            print("==============================")
            raise
        
        # This is required to get the lat/lon attributes
        lats, lons = self.process_coordinates(response.json())

        data = []
        for i, row in enumerate(response.json()['data']):

            # Add a step to turn the created_at string to a datetime object
            #  - this makes functionality easier to use later on.
            date = datetime.strptime(row['created_at'],'%Y-%m-%dT%H:%M:%S.%fZ')
            data.append([row['id'], row['author_id'], " ".join(row['text'].splitlines()), row['geo'], date, lats[i], lons[i]])

        while("next_token" in response.json()["meta"]):
            request_num = request_num + 1
            time.sleep(4)
            if(self.tweet_count is None):
                print("Making request %d of N" % request_num)
            else:
                print("Making request %d of %d" % (request_num, total_requests))

            payload["next_token"] = response.json()["meta"]["next_token"]
            response = requests.get("https://api.twitter.com/2/tweets/search/all", params=payload, headers=self.create_header())

            if (response.status_code != 200):
                print("==============================")
                print("The status is not 200. Response: ")
                print(response)
                print("Payload:")
                print(payload)
                print("==============================")
                raise

            lats, lons = self.process_coordinates(response.json())

            for i, row in enumerate(response.json()['data']):
                row['text'] = " ".join(row['text'].splitlines()) # remove newlines from text
                if('geo' in row):
                    data.append([row['id'], row['author_id'], row['text'], row['geo'], row['created_at'], lats[i], lons[i]])
                else:
                    data.append([row['id'], row['author_id'], row['text'], None , row['created_at'], lats[i], lons[i]])
        
        df = TwitterDataFrame(pd.DataFrame(data))
        df = df.rename(columns={
            0:'id',
            1:'author_id',
            2:'text',
            3:'geo',
            4:'created_at',
            5:'lat',
            6:'lon'
        })
        print("Query finished")
        return df

    def __str__(self):
#         print(self.queryString)
        
        return "query: "+self.queryString+"; time: "+self.startTime+" to "+self.endTime
        