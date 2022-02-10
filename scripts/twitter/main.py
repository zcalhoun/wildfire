from searchTwitter import TwitterSearchTerm
import pandas as pd
import time

# Construct list of months to cycle through
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
# EDIT HERE to change years.
years = ['2018','2019']#,'2020','2021']
all_months = [y+'-'+m for y in years for m in months]

# Last month
# The script stops when it hits this month, and does not get data from this month.
last_month = "2019-01"

# Construct query term
# words = [
#     "breathe",
#     "asthma",
#     "lungs",
#     "cough",
#     "headache",
#     "(itchy eyes)",
#     "(sore throat)"]
# EDIT here to change the query...
#query_term = "("+" OR ".join(words)+") -has:links -is:retweet has:geo -has:media place_country:us"

# In some cases, we do not want to query on ANY words.
# In this case, we want to pull tweets directly for San Francisco
query_term = "-has:links -is:retweet point_radius:[-122.5 37.8 25mi] has:geo lang:en -has:media place_country:us" 

# EDIT here for 
data_target_dir = '../../data/san_francisco/'

# You should not need to edit anything below here.
errors = []

for i in range(len(all_months)):
    
    print("Starting query for", all_months[i])
    searchQuery = TwitterSearchTerm(query_term, all_months[i]+"-01T00:00:00z", all_months[i+1]+"-01T00:00:00z")

    # Get the term count
    print("Querying for term count...")
    try:
        searchQuery.get_term_count()
        if(i==0):
            with open(data_target_dir+"check.txt","w") as f:
                f.write(searchQuery.tweet_count)
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
    # EDIT here to change file destination
    tweets.to_csv(data_target_dir+all_months[i]+'.csv', index=False)
    # Break if last month is this month
    print()
    # EDIT here to be the last month
    if (all_months[i+1] == last_month):
        print("Script complete")
        break
    else:
        time.sleep(1) # Sleep for a second before going to the next operation

# After for loop runs...print errors, if any.

if(len(errors) > 0):
    # EDIT here to change file destination
    with open(data_target_dir+'errors.txt', 'w') as f:
        for e in errors:
            f.write("%s\n" % e)
