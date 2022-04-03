from searchTwitter import TwitterSearchTerm
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, date, timedelta

SLEEP_TIME = 4
RETRY_LIMIT = 4
errors = []

def query_twitter(query):
    curr_date = query.start_date
    query_term = query.query_term()
    data_target_dir = query.output_dir

    total_tweets = 0

    while curr_date < query.end_date:
        print("Starting query for", curr_date)

        # Find upper bound of our tweet window
        curr_end_date = min(query.end_date, curr_date + timedelta(days=30))
        searchQuery = TwitterSearchTerm(query_term, str(curr_date)+"T00:00:00z", str(curr_end_date)+"T00:00:00z")

        # Get the term count
        print("Querying for term count...")
        for i in range(RETRY_LIMIT):
            try:
                searchQuery.get_term_count()
                if curr_date == query.start_date:
                    with open(data_target_dir+"check.txt","w") as f:
                        f.write(str(searchQuery.tweet_count))
            except Exception as e:
                print(e)
                errors.append(curr_date)
                time.sleep(SLEEP_TIME)
                continue
            else:
                break

        total_tweets += searchQuery.tweet_count

        # Retrieve the tweets
        print("Retrieving tweets")
        for i in range(RETRY_LIMIT):
            try:
                tweets = searchQuery.get_tweets()
            except Exception as e:
                print(e)
                errors.append(curr_date)
                time.sleep(SLEEP_TIME)
                continue
            else:
                break

        # Save tweets to a file.
        tweets.to_csv(data_target_dir+str(curr_date)+'.csv', index=False)

        # Sleep for a second before going to the next operation
        print()
        time.sleep(1)

    print("Script complete \n {} tweets collected \n {} failures".format(total_tweets, len(errors)))
    return

@dataclass
class TwitterQuery:
    output_dir: str
    lat: float
    lon: float
    start_date: date
    end_date: date

    def query_term(self) -> str:
        return "-has:links -is:retweet point_radius:[{} {} 25mi] has:geo lang:en -has:media place_country:us".format(self.lon, self.lat)

def main():
    parser = argparse.ArgumentParser(description="Geotagged Tweet Query Generator:")
    parser.add_argument("-o", '--output_dir', type=str, help="Enter the relative output directory", required=True)
    parser.add_argument("-c", "--coordinates", nargs='+', type=float, help="Enter latitude longitude", required=True)
    parser.add_argument("-d", "--dates", nargs="+", type=str, help="Enter startdate enddate (yyyy-mm-dd)", required=True)

    args = parser.parse_args()

    # Clean coordinates
    if len(args.coordinates) != 2:
        raise ValueError("Improper number of coordinates")

    lat, lon = args.coordinates
    if abs(lat) > 90 or abs(lon) > 180:
        raise ValueError("Improper coordinate values")

    # Clean dates
    if len(args.dates) != 2:
        raise ValueError("Improper number of dates")

    start_date, end_date = args.dates   
    try:
        start_date = datetime.strptime(start_date,'%Y-%m-%d').date()
        end_date = datetime.strptime(end_date,'%Y-%m-%d').date()
    except:
        raise ValueError("Improper date format")

    query = TwitterQuery(output_dir=args.output_dir, lat=lat, lon=lon, start_date=start_date, end_date=end_date)

    query_twitter(query)

    return

if __name__ == "__main__":
    main()

