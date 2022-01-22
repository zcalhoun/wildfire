# Twitter Search Term
This repository contains helpful functions along with a TwitterSearchTerm class to facilitate pulling large sets of data from Twitter.

## Goals of this repository
* Make searching twitter using the 2.0 API quick and easy.
* Create functionality that generates twitter files reliably and easily.

## Getting started
Currently, you only need the `utils.py` and `searchTwitter.py` file to get started. Once you have these downloaded on your machine, you will need to generate a bearer token.

The `utils.py` folder contains a function to make generating a bearer token easy. Once you generate that token, you are ready to start searching twitter.r

### Generating your bearer token
Twitter uses a bearer token to authenticate with their endpoint. This token should be stored in the file `.twitter_creds.json`. For security reasons, make sure you add this file to your `.gitignore` before pushing to a repository

*If you are new using a `.gitignore` file, you can use the following one liner in bash to generate that file properly (make sure you are in your working directory):*
```bash
echo ".twitter_creds.json" > .gitignore
```

To generate the your JSON file containing the bearer token, you'll need your API Key and your API Key Secret. You can then run the following code with your values inserted. If all goes well, you'll see the response "Token generated" and `True`. 

```python
import utils as ut
    
r = ut.generate_bearer_token("api_key_here", "api_key_secret_here")
print(r)
```

This function automatically creates the `.twitter_creds.json` file, which is referenced when making other calls to the Twitter API.

You are now ready to search Twitter!

## Making your first search

When making your first search query, I recommend the following process:
1. Generate your search term.
2. Run a "count" query to gauge how many tweets you should expect.
3. Once your are comfortable with the count, *then* execute the tweet search. By checking the count first, you'll ensure you won't accidentally hit the [twitter api rate limit](https://developer.twitter.com/en/docs/twitter-api/rate-limits), and that your query will take about as long as you'd expect.

Because twitter limits full-archive tweet searches to 1 request / sec, you should expect that your query takes a few seconds to run (each query gets a maximum of 500 results, so if your count is 10000, you should expect that your query takes 20 seconds to completely grab all of the data).