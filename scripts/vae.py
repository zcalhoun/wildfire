"""
    This script contains classes to handle tweet data and generate a
    VAE to reconstruct the count vectors
"""
import os
from datetime import datetime
import pandas as pd
import torch
import joblib
import numpy as np
from functools import lru_cache

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# For NLP
import spacy
from nltk.tokenize import TweetTokenizer


AQI_PATH = "../data/aqi_data/daily_aqi_by_county_2018.csv"
TWEET_PATH = "../data/san_francisco/"


class Tweets:
    """Tweets class. This class handles the data and preprocesses
    so that the data can be loaded easily into whatever format
    is needed.
    """

    def __init__(
        self,
        path,
        agg_count=1000,
        sample_rate=5,
        verbose=False,
        min_df=100,
        max_df=0.1,
        test_size=0.2,
        random_state=42,
    ):
        """
        Input:
            path: directory of twitter files, unprocessed.

            agg_count: the number of tweets to aggregate by.

            sample_rate: the number of total samples that we want
            to get for each day.

            verbose: whether to print out steps of loading the data.

            min_df: passed to the count_vec. This determines the amount
                of tweets that the word must appear in to be included.

            max_df: pass to the count_vec. This indicates the max
                number of tweets that the word can occur in.

            test_size: The percentage of tweets that are held out for
                the test set.

        This class should build a count vector from the tweets themselves,
        then store the tweets in an array that can be sampled from.

        """
        self.path = path
        self.test_size = test_size
        self.random_state = random_state

        # Load in each of the CSVs.
        print("Loading in the data...")
        tweets = self._load_data()

        # Remove values without date or tweet
        tweets = tweets.dropna()
        # Perform some preprocessing as an intermediate step
        # This is a very expensive line of code (takes a long time
        # and I am going to cache the results to use between runs.
        if cached(path, "lemmatized.joblib"):
            print(
                "Cached file was found...loading lemmatized", "tweets from the cache."
            )
            tweets["clean_tweets"] = load_cached(path, "lemmatized.joblib")
        else:
            print("No cache found. Loading now")
            tweets["clean_tweets"] = self._preprocess(tweets)

        tweets["date"] = [
            datetime.strptime(d, "%Y-%m-%d %H:%M:%S").date()
            for d in tweets["created_at"]
        ]

        # Create the count vector to process the tweets
        print("Creating the count vector")
        self.count_vec = CountVectorizer(
            stop_words="english", min_df=min_df, max_df=max_df
        )
        if test_size > 0:
            self.x, self.x_test = train_test_split(
                tweets, test_size=test_size, random_state=random_state
            )
        else:
            self.x = tweets

        # Create the count vectors
        x_cv = self.count_vec.fit_transform(self.x["clean_tweets"])
        # Remove unnecessary information and insert the count vector
        # into the x array
        self.x = np.array(list(zip(self.x["date"], x_cv)))

        # Save the cached count vector for future comparison
        save_to_cache(self.path, self.count_vec, "count_vec.joblib")

        if test_size > 0:
            x_test_cv = self.count_vec.transform(self.x_test["clean_tweets"])
            self.x_test = np.array(list(zip(self.x_test["date"], x_test_cv)))

        self.agg_count = agg_count
        self.sample_rate = sample_rate
        self.data = None
        self.vocab_size = len(self.count_vec.get_feature_names_out())

    def _load_data(self):
        """
        This function reads the files from the path
        and returns a concatenated version of the data.
        """
        data_frame = []
        # Load the data
        files = os.listdir(self.path)
        # If there is a cached file, then remove from
        # the list
        if "cached" in files:
            files.pop(files.index("cached"))

        for file in files:
            data_frame.append(pd.read_csv(self.path + file))

        return pd.concat(data_frame)

    def _preprocess(self, tweets):
        """
        This function is used to handle lemmatizing the data
        prior to its use.
        """
        tweet_tokenizer = TweetTokenizer()
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        lemmatized = []

        for tweet in tweets["text"]:
            clean_tweet = [
                w for w in tweet_tokenizer.tokenize(tweet.lower()) if w.isalpha()
            ]
            doc = nlp(" ".join(clean_tweet))
            lemmatized.append(" ".join([token.lemma_ for token in doc]))

        # Save this to the cached folder
        # Try to save the file to the cached folder.
        # If it doesn't exist...create the cache.
        try:
            joblib.dump(lemmatized, self.path + "cached/lemmatized.joblib")
        except FileNotFoundError:
            os.mkdir(self.path + "/cached/")
            joblib.dump(lemmatized, self.path + "cached/lemmatized.joblib")
        return lemmatized

    def load(self, test=False):
        """ This function handles loading the data from the count vector data."""
        if test:
            return TweetDataset(
                self.x_test, agg_count=self.agg_count, sample_rate=self.sample_rate
            )

        return TweetDataset(self.x, sample_rate=self.sample_rate)


class TweetDataset(Dataset):
    """This class converts the pandas dataframe into a tensor
        that will be loaded into the VAE"""

    def __init__(self, df, agg_count=1000, sample_rate=5, random_state=42):
        """
            Inputs:
                df - the dataframe object with the "count_vec" column and the date column.
                acc_count - the number of tweets to aggregate by
                sample_rate - the number of times to sample each day
        """
        # Define the objects used in the two functions below
        self.dates = list(set(df[:, 0]))
        self.agg_count = agg_count
        self.sample_rate = sample_rate
        self.generator = np.random.default_rng(seed=random_state)
        self.df = df

        # Added to support looking up aqi
        self.aqi = load_aqi()

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.dates) * self.sample_rate

    @lru_cache(maxsize=None)  # Save results of this function for quicker access
    def __getitem__(self, idx):
        """
        This function selects the date at the index
        provided. If the index is greater than the length
        of the array (i.e., we are sampling multiple examples
        from a date), then wraparound and keep sampling.
        """
        # Select the date
        date = self.dates[idx % len(self.dates)]
        # Randomly sample from this date.
        #  1. Only look at count_vecs on this date.
        #  2. Sample agg_count number of tweets, sum the count vectors,
        #     and return

        count_vecs = self.df[np.where(self.df[:, 0] == date)][:, 1]

        # Sample using the generator
        sample = self.generator.choice(count_vecs, self.agg_count, replace=True)

        # Load the aqi to return
        # Return the numpy array, summed along its axis.
        return (
            torch.from_numpy(sample.sum().toarray()).float().requires_grad_(False),
            torch.tensor(self.aqi.get(date)),
        )


def load_aqi():
    """
    This function returns a dictionary containing AQI with the 
    date object as the keys.
    """
    df = pd.read_csv(AQI_PATH)

    df = df[
        (df["State Name"] == "California") & (df["county Name"] == "San Francisco")
    ][["Date", "AQI"]]

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").apply(datetime.date)

    df["AQI"] = np.log10(df["AQI"])

    return df.set_index("Date").to_dict().get("AQI")


class VAE(nn.Module):
    """
    Should rename -- PFA for Poisson Factor Analysis
    """

    def __init__(self, vocab, num_components=20, prior_mean=0, prior_logvar=0):
        """
        Inputs
        --------
        vocab<int>: the size of the vocabulary

        This model only has the variational layer, then the output
        to the reconstruction. At this point, there are no hidden layers.
        """
        super().__init__()
        self.num_components = num_components

        self.prior_mean = torch.tensor(prior_mean)
        self.prior_logvar = torch.tensor(prior_logvar)

        self.enc_logvar = nn.Linear(vocab, num_components, bias=False)
        self.enc_mu = nn.Linear(vocab, num_components, bias=False)
        self.W_tilde = torch.rand(num_components, vocab, requires_grad=True)
        self.pois_nll = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()

        self.beta = nn.Linear(1, 1, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)

        s_tilde = self.reparameterize(mu, logvar)

        s = self.softplus(s_tilde)
        W = self.softplus(self.W_tilde)

        # Predict y using the first node from s
        y_hat = self.beta(s[:, :, 1])

        return s, W, mu, logvar, y_hat

    def get_topic_dist(self, x):
        """
        When it comes to looking at the norm, we want to calculate the
        probability that a certain sample belongs to each topic.
        """
        s, _ = self.encode(x)
        W = self.parameters()  # TODO - figure out which parameters to add.
        norm = torch.norm(s @ W, p=1)  # Return the L1 norm
        # TODO -- add in the multinomial distribution.

        # TODO - need to calculate elementwise product.
        return s @ W / norm

    def _kl_divergence(self, mean, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # BUT...
        # Code extended to handle a more informative prior
        # Referencing this derivation found here:
        # https://stanford.edu/~jduchi/projects/general_notes.pdf
        # Assume diagonal matrices for variance
        KLD = -0.5 * torch.sum(
            1
            + logvar
            - self.prior_logvar
            - (mean - self.prior_mean) ** 2 / self.prior_logvar.exp()
            - logvar.exp() / self.prior_logvar.exp()
        )

        return KLD

    def loss_function(self, recon_x, x, mu, logvar, y, y_hat):
        KLD = self._kl_divergence(mu, logvar)
        PNLL = self.pois_nll(recon_x, x)
        # This will disproportionately weight higher values of y
        MSE = (y - y_hat).pow(10).mean()
        return PNLL, MSE, KLD

    @torch.no_grad()
    def reconstruct(self, X):
        s, W, mu, logvar = self.forward(X)

        return s @ W


def cached(path, doc_type):
    """
    This function looks for the path in the list of cached
    objects and returns true if the line exists."""
    files = os.listdir(path)
    if "cached" in files:
        cached_files = os.listdir(path + "cached/")
        if doc_type in cached_files:
            return True
    return False


def load_cached(path, doc_type):
    """This function loads cached data, assuming
        it exists. This data is return as it was
        saved in the file."""
    return joblib.load(path + "cached/" + doc_type)


def save_to_cache(path, doc, file_name):
    """This saves a document to the a cache"""
    joblib.dump(doc, path + "cached/" + file_name)


if __name__ == "__main__":
    print("Begin testing")

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Set up the tweets module
    # Cache this output for future use
    if not cached(TWEET_PATH, "tweets_maxdf005_agg1000_sample10.joblib"):
        tweets = Tweets(TWEET_PATH, max_df=0.005, agg_count=1000, sample_rate=10)
        save_to_cache(TWEET_PATH, tweets, "tweets_maxdf005_agg1000_sample10.joblib")
    else:
        tweets = load_cached(TWEET_PATH, "tweets_maxdf005_agg1000_sample10.joblib")
    # Load the train and test data
    print("Loading the training and test data.")
    x_train = tweets.load(test=False)
    x_test = tweets.load(test=True)

    train_loader = DataLoader(x_train, batch_size=128)
    test_loader = DataLoader(x_test, batch_size=128)

    model = VAE(tweets.vocab_size, num_components=100, prior_mean=1)

    model.to(device)

    EPOCHS = 30
    print_rate = 10
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run training and testing on the model.
    loss_results = {
        "train": {"pnll": [], "mse": [], "kld": [], "total": []},
        "val": {"pnll": [], "mse": [], "kld": [], "total": []},
    }
    for epoch in range(EPOCHS):
        epoch_train_loss = 0
        epoch_test_loss = 0
        # Run method on training
        model.train()
        avg_pnll = 0
        avg_mse = 0
        avg_kld = 0
        for batch_idx, (data, y) in enumerate(train_loader):
            # Add training data to GPU
            data = data.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            s, W, mu, logvar, y_hat = model(data)
            s = s.to(device)
            W = W.to(device)
            y_hat = y_hat.to(device)
            recon_batch = s @ W  # Calculate the reconstructed matrix
            recon_batch = recon_batch.to(device)
            mu = mu.to(device)
            logvar = logvar.to(device)
            PNLL, MSE, KLD = model.loss_function(
                recon_batch, data, mu, logvar, y, y_hat
            )
			
            loss = torch.mean(PNLL + MSE + 0.1 * KLD)
            loss.backward()
            optimizer.step()
            avg_pnll += torch.mean(PNLL).item()
            avg_mse += torch.mean(MSE).item()
            avg_kld += torch.mean(KLD).item()
            epoch_train_loss += loss.item()
            
            if batch_idx % print_rate == 0:
                print(
                    "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item() / len(data),
                    )
                )
        print(
            "===> Epoch: {} Average Loss: {:.4f}".format(
                epoch, epoch_train_loss / len(train_loader.dataset)
            )
        )
        loss_results["train"]["pnll"].append(avg_pnll / len(train_loader.dataset))
        loss_results["train"]["mse"].append(avg_mse / len(train_loader.dataset))
        loss_results["train"]["kld"].append(avg_kld / len(train_loader.dataset))
        loss_results["train"]["total"].append(epoch_train_loss / len(train_loader.dataset))

        # loss['train'].append(epoch_train_loss)

        # Capture testing performance.
        model.eval()
        frobenius_norms = []
        poisson = []
        mean_squared_error = []
        kl_divergence = []
        avg_pnll = 0
        avg_mse = 0
        avg_kld = 0
        with torch.no_grad():
            for batch_idx, (data, y) in enumerate(test_loader):
                # Add to GPU
                data = data.to(device)
                y = y.to(device)
                s, W, mu, logvar, y_hat = model(data)
                recon_batch = s @ W
                PNLL, MSE, KLD = model.loss_function(
                    recon_batch, data, mu, logvar, y, y_hat
                )
                loss = torch.mean(PNLL + MSE + 0.1 * KLD)
                avg_pnll += torch.mean(PNLL).item()
                avg_mse += torch.mean(MSE).item()
                avg_kld += torch.mean(KLD).item()
                epoch_test_loss += loss.item()

                # Calculate frobenius norm of the reconstructed matrix
                frobenius_norms.append(
                    torch.norm(recon_batch - data, p="fro", dim=2).mean().item()
                )
                mean_squared_error.append(MSE)
                kl_divergence.append(KLD)

        avg_mse /= len(test_loader.dataset)
        avg_pnll /= len(test_loader.dataset)
        avg_kld /= len(test_loader.dataset)
        epoch_test_loss /= len(test_loader.dataset)

        # Append results to the json
        loss_results["val"]["pnll"].append(avg_pnll)
        loss_results["val"]["mse"].append(avg_mse)
        loss_results["val"]["kld"].append(avg_kld)
        loss_results["val"]["total"].append(epoch_test_loss)

        avg_f_norm = sum(frobenius_norms) / len(frobenius_norms)
        print("===> Test set loss: {:.4f}".format(epoch_test_loss))
        # Print frobenius norm
        print("======> Test set frobenius norm: {:.4f}".format(avg_f_norm))
        print("======> Test set mean squared error: {:.4f}".format(avg_mse))
        print("======> Test set kl divergence: {:.4f}".format(avg_kl))
        print("======> Test set poisson: {:.4f}".format(avg_poisson))
    torch.save(model.state_dict(), "./model/model_3epoch.pt")

    # Save the loss to a file
    with open("./model/loss_3epoch.json", "w") as f:
        json.dump(loss, f)

