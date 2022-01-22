from searchTwitter import TwitterSearchTerm
import pandas as pd
import time

# Construct list of months to cycle through
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
years = ['2018','2019','2020','2021']
all_months = [y+'-'+m for y in years for m in months]

# Construct query term
words = [
    "breathe",
    "asthma",
    "lungs",
    "cough",
    "headache",
    "(itchy eyes)",
    "(sore throat)"]

query_term = "("+" OR ".join(words)+") -has:links -is:retweet has:geo -has:media place_country:us"

# test_term = "\"air pollution\" -has:links -is:retweet has:geo -has:media place_country:us"
errors = []
for i in range(len(all_months)):
    
    print("Starting query for", all_months[i])
    searchQuery = TwitterSearchTerm(query_term, all_months[i]+"-01T00:00:00z", all_months[i+1]+"-01T00:00:00z")

    # Get the term count
    print("Querying for term count...")
    try:
        searchQuery.get_term_count()
    except:
        errors.append(all_months[i])
        continue
    # Retrieve the tweets
    print("Retrieving tweets")
    try:
        tweets = searchQuery.get_tweets()
    except:
        errors.append(all_months[i])
        continue
    # Save tweets to a file.
    tweets.to_csv('../data/health_tweets/'+all_months[i]+'.csv', index=False)
    # Break if last month is this month
    print()
    if (all_months[i+1] == "2021-10"):
        print("Script complete")
        break
    else:
        time.sleep(1) # Sleep for a second before going to the next operation

# After for loop runs...print errors, if any.

if(len(errors) > 0):
    with open('../data/health_tweets/errors.txt', 'w') as f:
        for e in errors:
            f.write("%s\n" % e)
