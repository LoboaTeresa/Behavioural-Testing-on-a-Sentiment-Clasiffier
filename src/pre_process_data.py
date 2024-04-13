import os
import pickle
import random
import re
import string
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer

# Define stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stpwrd = stopwords.words('english')
stpwrd.extend(string.punctuation)  # add punctuation symbols to stopwords: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
stpwrd.extend(['rt', 'co', 'https', 'http', 'amp', 'us'])  # add common twitter terms to stopwords
commwords = ['climate', 'change', 'global', 'warming','like', 'ðÿ', 'il', 'le', 'gt']
stpwrd.extend(commwords) # add common words to stopwords

lem = WordNetLemmatizer()

def regex_clean(txt: str, regex: str):
    """Remove any text matching a RegEx pattern.

    Args:
        txt (str): A text string that you want to parse and remove matches.
        regex (str): A text string of the regex pattern you want to match.

    Returns:
        str: The same txt string with the matches removes
    """
    return " ".join(re.sub(regex, " ", txt).split())

def remove_stpwrds(text: str):
    """Clean up a tweet.

    First, the function will make the text lowercase, remove mentions, links, punctuation, numbers, and single characters.
    Then, it will tokenize the text and remove stopwords. Finally, it will lemmatize the words.

    Args:
        text (str): A tweet that you want to parse and remove matches

    Returns:
        str: Cleaned up string - tokenized & stemmed!
    """
    # 1. Pre Token Cleaning
    text = text.lower()  # make lowercase
    text = regex_clean(text, r'(?:)@[A-Za-z0-9\-\.\_]+(?:)')  # remove mentions
    text = regex_clean(text, r'(?:(?:http?|https?|ftp):\/\/)\S+')  # remove links
    text = regex_clean(text, r"[^\w\s]" )  # remove punctuation
    text = regex_clean(text, r"\d")  # remove numbers
    text = regex_clean(text, r"\b\w\b")  # remove single characters
    text = regex_clean(text, r"(\S+|\b)[^a-zA-Z\s]\S+")  # remove non-alphabetic chars and non-english words
    text = regex_clean(text, r"[^a-z]+")  # remove more non-alphabetic chars and non-english words

    # 2. Tokenization
    text = word_tokenize(text)
    
    # 3. Token cleaning
    text = [lem.lemmatize(word) for word in text if word not in stpwrd] # remove stopwords and lemmatize
    
    return text


def df_cleaner(df: pd.DataFrame):
    """Clean and summarize a DataFrame containing tweet data.
    
    It will filter out rows with no words in the tweet text and provide some summary statistics about the dataset. 

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    word_mask = df['token_len'] != 0
    words = df.loc[word_mask]

    return words

def untokener(tokens: list):
    """Stitch tokens back together into a string for TF-IDF Vectorizer.

    TF-IDF Vectorizer requires a string input, so this function will join the tokens back together into a string.
    The TF-IDF Vectorizer will then tokenize the string again.

    Args:
        tokens (list): list of tokens (strings) from a tweet.

    Returns:
        str: strink of tokens joined by a space.
    """
    return ' '.join(tokens)

def get_token_buckets(df: pd.DataFrame):
    """Create token buckets based on word length ranges in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing tweet data.

    Returns:
        dict: A dictionary where keys are tuples representing word length ranges
        and values are lists of tokens extracted from tweets falling within
        each corresponding range.
    """
    word_ranges = [(1, 6), (6, 11), (11, 16), (16,50)]  # High cap on last one to be more of a 16+ category
    buck_dict = {}

    for limits in word_ranges:
        lim_df = df[(df.token_len >= limits[0]) & (df.token_len < limits[1])]
        s = ' '.join(lim_df['clean_message'])
        lim_tokens = word_tokenize(s) 
        buck_dict[limits] = lim_tokens
        
    return buck_dict

def artificial_tweet(to_len, buckets):
    """Generate an artificial tweet of a specified length using token buckets.

    Args:
        to_len (int): The desired length of the artificial tweet in terms of the number of tokens.
        buckets (dict): A dictionary where keys are tuples representing word length ranges
            and values are lists of tokens extracted from tweets falling within each corresponding range.

    Returns:
        list: A list of tokens forming the artificial tweet.
    """
    for lims in buckets.keys():
        if to_len >= lims[0] and to_len < lims[1]:
            curr_buck = buckets[lims]

    return random.choices(curr_buck, k = to_len)

##Distribution function
def get_len_dist(cdf):
    """Get the distribution of tweet lengths from a DataFrame.

    Args:
        cdf (pd.DataFrame): The DataFrame containing tweet data.

    Returns:
        list: A list containing the lengths of tweets in terms of the number of tokens.
    """
    return list(cdf.token_len)

def upsample_tweets(raw_df: pd.DataFrame, nu_len: int) -> pd.DataFrame:
    """Upsample the tweet data to a specified length.

    Args:
        raw_df (pd.DataFrame): The original DataFrame containing tweet data.
        nu_len (int): The desired length of the upsampled DataFrame.

    Returns:
        pd.DataFrame: The upsampled DataFrame with additional synthetic tweets.
    """
    assert(raw_df.shape[0] < nu_len), "this won't work if the new length is shorter than the original!"
    assert(raw_df.target.min() == raw_df.target.max()), "this only works on data that is labelled as the same - pls filter"
    
    # Confirm df label
    label = raw_df.target.min()
    
    df = raw_df.copy()
    df['is_authentic'] = 1  # Create a column that tracks the authenticity of the tweet
        
    # First - get lengths of all tweets - to randomly select from - distribution of lengths
    lengths = get_len_dist(df)

    # Then we generate our buckets
    bucket_dict = get_token_buckets(df)
    
    # Now we have our two components - now we can construct a series of artifical tweets and append to the data frame
    #- upsample!
    ###new length 
    to_add = nu_len - df.shape[0]
    
    for __ in range(to_add):
        fake_len = random.choice(lengths)  # Randomly select a length for the new tweet from the dist
        fake_tokens = artificial_tweet(fake_len, bucket_dict)  # Generate the tokens for the new tweet
        fake_retoke = untokener(fake_tokens)        

        app_dict = {'message': 'SYNTHETIC DATA',
                   'tweetid' : np.nan,
                   'target' : label,
                   'tokens' : fake_tokens,
                   'token_len' : fake_len,
                   'clean_message' : fake_retoke,
                   'is_authentic' : 0}

        df = df._append(app_dict, ignore_index = True)

    return df

def pre_process_data(data_df: pd.DataFrame, 
                     data_type: str = 'custom'
                     ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Preprocess data for the model.

    Args:
        data (pd.DataFrame): data to run inference on.
        data_type (str, optional): specify if the data is training,
            val or test data or if it is "custom", which means it is data
            you want to run inference on. Defaults to 'custom'. Posible 
            values are: 'train', 'val', 'test', 'custom'.
    Returns:
        Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]: A tuple containing the cleaned DataFrame,
        feature matrix, and optional target vector.
    """
    main_directory = str(Path().resolve().parent)
    print("main_dir:", main_directory)
    
    data_df['tokens'] = data_df['message'].apply(remove_stpwrds)
    data_df['token_len'] = data_df['tokens'].apply(len)

    clean_df = df_cleaner(data_df.copy())
    clean_df['clean_message'] = clean_df['tokens'].apply(untokener)

    if data_type == 'train' and 'target' in clean_df.columns:
        # Resampling
        ## Separate classes
        resamp = clean_df.copy()

        resamp_anti = resamp[resamp['target'] == 0]
        resamp_neut = resamp[resamp['target'] == 1]
        resamp_pro = resamp[resamp['target'] == 2]
        resamp_news = resamp[resamp['target'] == 3]

        ## Downsample majority class
        resamp_pro = resample(resamp_pro, replace = False, n_samples = 10000)
        
        ## Upsample minority class
        upped_dict = {}
        for minor, key in [(resamp_anti,'anti'), (resamp_neut, 'neut'), (resamp_news,'news')]:
            upped_dict[key] = upsample_tweets(minor, 10000)

        resamp_pro['is_authentic'] = 1
        frames = [resamp_pro, upped_dict['anti'], upped_dict['neut'], upped_dict['news']]

        ## Stitch them back together
        clean_df = pd.concat(frames)

    # Vectorize
    tfidf = TfidfVectorizer(analyzer = 'word',
                            max_features = 1500,
                            min_df = 1)

    if data_type == 'train':
        tr_feats = tfidf.fit_transform(clean_df['clean_message'])
        # Save the TfidfVectorizer object to a file
        # if path exists
        save_path = main_directory + '/data/tfidf_vectorizer.pkl'
        if not os.path.exists(save_path):
            save_path = "./data/tfidf_vectorizer.pkl"
    
        with open(save_path, 'wb') as f:
            pickle.dump(tfidf, f)

    else:
        # Load the saved TfidfVectorizer object from the file
        # check if is file
        vector_path = main_directory + "/data/tfidf_vectorizer.pkl"
        if not os.path.isfile(vector_path):
            vector_path = "data/tfidf_vectorizer.pkl"
        with open(vector_path, 'rb') as f:
            tfidf = pickle.load(f)
        # Use the loaded TfidfVectorizer object to transform the individual tweet
        tr_feats = tfidf.transform(clean_df['clean_message'])

    data_y = clean_df['target'].values if 'target' in clean_df.columns else None
    data_X = tr_feats

    return clean_df, data_X, data_y
