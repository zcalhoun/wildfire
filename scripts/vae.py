import pandas as pd
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# For NLP
import spacy
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

class TweetData():
	def __init__(self, test_size=0.2, debug=False, min_df=5, max_df=0.5):
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
		if(debug):
			self.path = '../data/san_francisco/2018-02.csv'
		else:
			self.path = 'undefined'

		# Define
		self.test_size = 0.2
		self.count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)

	def get_tweet_count_vecs(self):
		# Load the data
		# Retrieve the file
		data = pd.read_csv(self.path)

		# If debug, only keep the first 100 tweets
		if(self.debug):
			data = data.iloc[0:100]
		
		# Clean the text
		tt = TweetTokenizer()
		nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
		lemmatized = []
		for tweet in data['text']:
			t = tweet.lower()
			t = tt.tokenize(t)
			t = [w for w in t if w.isalpha()]
			doc = nlp(" ".join(t))
			lemmatized.append(" ".join([token.lemma_ for token in doc]))
		
		data['lemmatized'] = lemmatized
		# Split into train/test splits
		X_train, X_test = train_test_split(lemmatized, test_size=self.test_size, random_state=42)
		
		# Apply count vectorizer
		self.X_train = self.count_vectorizer.fit_transform(X_train)
		self.X_test = self.count_vectorizer.transform(X_test)

		# Return the train/test count vectors
		return self.X_train, self.X_test

	def get_count_vec(self):
		return self.count_vectorizer


class TweetDataLoader():
	def __init__(self, test=False):
		"""
		Inputs
		-----------
		test<bool> : If test, then only import a subset of data. This is for
			debugging on my local machine.

		Outputs:
		-----------
		A data loader for handling tweets in the desired format.
		"""
		self.test = test
		# If for testing purposes, only load one example
		# csv for testing out functionality.
		if(test):
			self.path = '../data/san_francisco/2018-02.csv'



# Handle any arguments passed at runtime.
class VAE(nn.Module):

	def __init__(self):
		super(VAE, self).__init__()

		# VAE weights
		self.fc1 = nn.Linear(VOCAB, 400)
		self.fc21 = nn.Linear(400, 20)
		self.fc22 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(20,400)
		self.fc4 = nn.Linear(400, VOCAB)

		# Prediction weights
		self.fc_aqi = nn.Linear(20, 1)
	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

	def predict_aqi(self, mu):
		val = self.fc5(mu)
		return val

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, 784))
		z = self.reparameterize(mu, logvar)
		aqi = self.predict_aqi(mu)
		return self.decode(z), aqi, mu, logvar

	@torch.no_grad()
	def predict(self, x):
		'''
		Handles predicting AQI for any vector of tweets.
		'''
		x = torch.Tensor(x).to(device)
		a, aqi, b, c = self.forward(x)
		return aqi


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence + MSE summed over all
# elements and batch
def loss_function(recon_x, x, pred_aqi, aqi, mu, logvar):
	BCE = F.binary_cross_entropy(recon_x, reduction='sum')

	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	MSE = torch.mean((pred_aqi-aqi).pow(2))

	return BCE + KLD + MSE

def train(epoch):
	model.train()
	train_loss = 0
	for batch_idx, (data, aqi) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		recon_batch, pred_aqi, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, pred_aqi, aqi, mu, logvar)
		loss.backward()
		train_loss +=loss.item()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    # Running testing to validate model.
    print("Testing beginning")

    # Load training data
    # Custom data loader required.

    # Initialize model
