"""
    This script contains classes to handle tweet data and generate a
    VAE to reconstruct the count vectors
"""
import os
from datetime import datetime
import pandas as pd
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# For NLP
import spacy
from nltk.tokenize import TweetTokenizer


class Tweets():
    """Tweets class. This class handles the data and preprocesses
    so that the data can be loaded easily into whatever format
    is needed.
    """
    def __init__(self, path, agg_count=1000, sample_rate=5,
                 verbose=False, min_df=0.01, max_df=0.1, test_size=0.2,
                 random_state=42):
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
        tweets = self._load_data()

        # Perform some preprocessing as an intermediate step
        tweets['clean_tweets'] = self._preprocess()

        # Create the count vector to process the tweets
        self.count_vec = CountVectorizer(stop_words='english',
                                         min_df=min_df,
                                         max_df=max_df)
        if test_size > 0:
            self.x, self.x_test = train_test_split(tweets,
                                                   test_size=test_size,
                                                   random_state=random_state)
        else:
            self.x = tweets

        self.x['count_vec'] = self.count_vec.fit_transform(self.x['clean_tweets'])

        if test_size > 0:
            self.x_test['count_vec'] = self.count_vec.transform(self.x_test['count_vec'])

        self.agg_count = agg_count
        self.sample_rate = sample_rate
        self.data = None
        self.vocab_size = len(self.count_vec.get_feature_names_out)

    def _load_data(self):
        """
        This function reads the files from the path
        and returns a concatenated version of the data.
        """
        data_frame = []
        # Load the data
        for file in os.listdir(self.path):
            data_frame.append(pd.read_csv(self.path+file))

        return pd.concat(data_frame)

    def _preprocess(self):
        """
        This function is used to handle lemmatizing the data
        prior to its use.
        """
        tweet_tokenizer = TweetTokenizer()
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        lemmatized = []
        for tweet in self.tweets['text']:
            clean_tweet = [w for w in tweet_tokenizer.tokenize(tweet.lower())
                           if w.isalpha()]
            doc = nlp(" ".join(clean_tweet))
            lemmatized.append(" ".join([token.lemma_ for token in doc]))
        return lemmatized

    def load(self, test=False):
        """ This function handles loading the data from the count vector data."""
        if test:
            return TweetDataset(self.x_test)

        return TweetDataset(self.x)



class TweetDataset(Dataset):
    """This class converts the pandas dataframe into a tensor
       that will be loaded into the VAE"""

    def __init__(self, df, agg_count=1000, sample_rate=5):
        """
           Inputs:
               df - the dataframe object with the "count_vec" column and the date column.
               acc_count - the number of tweets to aggregate by
               sample_rate - the number of times to sample each day
        """
        # Set up the passed dataframe to ensure its dates are correct.
        df['date'] = [datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in
                    df['created_at']]

        # Define the objects used in the two functions below
        self.dates = list(set(df['date']))
        self.agg_count = agg_count
        self.sample_rate = sample_rate
        self.df = df

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.dates)*self.sample_rate

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
        #  2. Select 10,000 and sum.
        count_vecs = self.df[self.df['date'] == date]['count_vec']

        # Return a tensorized version of the array
        return torch.from_numpy(count_vecs.sample(self.agg_count).sum())


class TweetData():
    model = VAE()    """
    This class handles parsing out the tweet data
    """
    def __init__(self, test_size=0.2, debug=False, min_df=5, max_df=0.5, agg_count=1000):
        """
            Inputs
        -----------
        debug<bool> : If test, then only import a subset of data. This is for
            debugging on my local machine.

        test_size<float> : Determine the train/test split to be
            used when training the model.

        Outputs:
        -----------
        train_data<array> : an array of the train data
        test_data<array> : an array of the test data

        A data loader for handling tweets in the desired format.
        """
        self.debug = debug

        if debug==True:
            self.path = '../data/san_francisco/2018-02.csv'
        else:
            self.path = 'undefined'

        # Define
        self.test_size = 0.2
        self.count_vectorizer = CountVectorizer(stop_words='english',
                                                min_df=min_df, max_df=max_df)
        self.agg_count = agg_count

    def get_tweet_count_vecs(self):
        """
        This function loads the tweet data and converts it into a count vector.
        """


        # Load the data
        # Retrieve the file
        data = pd.read_csv(self.path)

        # If debug, only keep the first 100 tweets
        if(self.debug):
            data = data.iloc[0:10000]

        data = self.preprocess(data)

        X_train, X_test = train_test_split(data,
                                           test_size=self.test_size,
                                           random_state=42)

        # Apply count vectorizer
        self.X_train = self.count_vectorizer.fit_transform(X_train)
        self.X_test = self.count_vectorizer.transform(X_test)

        # Return the train/test count vectors
        return self.X_train, self.X_test

    def get_vocab_size(self):
        return self.X_train.shape[1]

    def preprocess(self, data):
        tt = TweetTokenizer()
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        lemmatized = []
        for tweet in data['text']:
            t = tweet.lower()
            t = tt.tokenize(t)
            # Removes hashtags and @s
            t = [w for w in t if w.isalpha()]
            doc = nlp(" ".join(t))
            lemmatized.append(" ".join([token.lemma_ for token in doc]))

        return lemmatized

    def to_tensor_dataset(self):
        x_train = TensorDataset(torch.Tensor(self.X_train.todense()))
        x_test = TensorDataset(torch.Tensor(self.X_test.todense()))
        return x_train, x_test

    def get_count_vec(self):
        return self.count_vectorizer

    def inverse_transform_count_vec(self, count_vec):
        # TODO - make this handle the CSR data format.
        # if(type(count_vec) == torch.utils.data.dataset.TensorDataset):
        return self.count_vectorizer.inverse_transform(count_vec)

    EPSILON = 1e-8


class VAE(nn.Module):
    """
    Should rename -- PFA for Poisson Factor Analysis
    """

    def __init__(self, vocab, num_components=20, prior_mean=0, prior_var=1):
        """
        Inputs
        --------
        vocab<int>: the size of the vocabulary

        This model only has the variational layer, then the output
        to the reconstruction. At this point, there are no hidden layers.
        """
        super(VAE, self).__init__()
        self.num_components = num_components

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.enc_mu = nn.Linear(vocab, num_components, bias=False)
        self.enc_logvar = nn.Linear(vocab, num_components, bias=False)
        self.W_tilde = torch.rand(num_components, vocab)
        self.pois_nll = nn.PoissonNLLLoss()
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)

        s_tilde = self.reparameterize(mu, logvar)

        s = self.softplus(s_tilde)
        W = self.softplus(self.W_tilde)

        return s, W, mu, logvar

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
        KLD = -0.5 * torch.sum(1 + logvar - torch.log(torch.Tensor(self.prior_var))
            - self.prior_var*((self.prior_mean-mu).pow(2) - logvar.exp()))

        return KLD

    def loss_function(self, recon_x, x, mu, logvar):
        KLD = self._kl_divergence(mu, logvar)
        PNLL = self.pois_nll(x, recon_x)
        return torch.mean(PNLL + KLD)

    def fit(self, X, n_epochs=20, lr=1e-3, print_rate=10):
        """
        Fit the model to the data, X. Assume X is in count vector format as a tensor.
        """
        train_loader = DataLoader(X, batch_size=128)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            epoch_train_loss = 0
            epoch_test_loss = 0
            for batch_idx, data in enumerate(train_loader):
                self.train()
                optimizer.zero_grad()
                s, W, mu, logvar = self.forward(data)
                with torch.no_grad():
                    recon_batch = s @ W # Calculate the reconstructed matrix
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                epoch_train_loss += loss.item()
                optimizer.step()
                if batch_idx % print_rate == 0:
                    print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data)))
            print('===> Epoch: {} Average Loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)
            ))

    @torch.no_grad()
    def reconstruct(self, X):
        s, W, mu, logvar = self.forward(X)

        return s @ W

if (__name__ == "__main__"):
    print("Begin testing")

    # Set up the tweets module
    tweets = Tweets('../data/test/')

    # Load the train and test data
    x_train = tweets.load(test=False)
    x_test = tweets.load(test=True)


    # Testing below this line soon...
    break
    train_loader = DataLoader(x_train, batch_size=10, shuffle=True)
    test_loader = DataLoader(x_test, batch_size=10, shuffle=False)

    # Initialize model
    model = VAE(tweets.vocab_size)
    # Train model

    # Test model
