import argparse
import logging
from src.utils import DataHandler

# Set up arguments
parser = argparse.ArgumentParser(description="Implementation of S-PFA")
###############################################################################
### Data parameters
###############################################################################
parser.add_argument(
    "--lemmatized_path", type=str, default="data/", help="path to lemmatized files"
)
parser.add_argument(
    "--target_path", type=str, default="data/storage", help="path to save loaded data"
)
parser.add_argument(
    "--sample_method", type=str, default="by_file", help="sample method"
)
parser.add_argument(
    "--tweet_agg_num", type=int, default=1000, help="number of tweets to aggregate"
)
parser.add_argument(
    "--tweet_sample_count",
    type=int,
    default=1,
    help="number of times to sample from day",
)
parser.add_argument(
    "--min_df", type=int, default=300, help="minimum document frequency"
)
parser.add_argument(
    "--max_df", type=float, default=0.01, help="maximum document frequency"
)
parser.add_argument(
    "--train_cities",
    nargs="+",
    help="Cities to include in the training set",
    required=True,
)
parser.add_argument(
    "--test_cities", nargs="+", help="Cities to include in the test set", required=True
)

###############################################################################
### Model parameters
###############################################################################


def main():
    global args
    args = parser.parse_args()

    # Create the count vector
    logging.info(args)
    data_handler = DataHandler(
        args.lemmatized_path,
        args.target_path,
        args.sample_method,
        args.tweet_agg_num,
        args.tweet_sample_count,
        args.min_df,
        args.max_df,
        args.train_cities,
        args.test_cities
    )

    # Load the data
    train_dataset = data_handler.create_train_dataset()
    test_dataset = data_handler.create_test_dataset()


if __name__ == "__main__":
    logging.basicConfig(filename="1000_10_by_file.log", level=logging.DEBUG)
    main()
