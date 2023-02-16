#
"""
subject: NLP HW3 - Part 1
autor: Amir Hussein Khaniki
"""
#

#
"""
Init Section
"""
#

# Libraries
import os
import re
import random
import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter as ngram_counter
from regex import match as re_match
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams, bigrams, trigrams, stem as stemmer, WordNetLemmatizer as lemmatizer, lm, FreqDist, LaplaceProbDist, SimpleGoodTuringProbDist
import keras
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as sc
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Functions

default_corpus = ''

# remove stopwords function
def remove_stopwords(text, array = False):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text if array else ' '.join(filtered_text)

# stem words in the list of tokenized words
def stem_words(text):
    ps = stemmer.PorterStemmer()
    lcs = stemmer.LancasterStemmer()
    word_tokens = word_tokenize(text)
    stems = [ps.stem(word) for word in word_tokens]
    return stems

# stem words in the list of tokenized words
def stem(text) :
    ps = stemmer.PorterStemmer()
    result = []   
    for i in text :
        result.append(ps.stem(i))
        
    return ' '.join(result)

# lemmatize string
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return lemmas

# convert number into word
"""
def convert_number(text):
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []
 
    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
 
        # append the word as it is
        else:
            new_string.append(word)
 
    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str
    """

""" 
wst = tokenize.WhitespaceTokenizer()
pst = tokenize.PunktSentenceTokenizer()
twt = tokenize.TreebankWordTokenizer()
tokenized_result = wst.tokenize(content)
processed_result = ' '.join(tokenized_result)
list_of_sentences = pst.tokenize(processed_result)
list_of_tokens_tree = twt.tokenize(processed_result)
regex_tokenizer = tokenize.RegexpTokenizer(r"\w+", gaps=True)
list_token_Regexp=regex_tokenizer.tokenize(' '.join(list_of_tokens_tree))
"""

# content preprocessing
def pre_process(content):
    processed_content = content.lower() #=> convert text to lowercase
    processed_content = processed_content.translate(str.maketrans('', '', string.punctuation)) #=> remove punctuation
    #processed_content = re.sub(r'[^\w\s]', '', processed_content) #=> remove anything base on regex
    processed_content = ' '.join(processed_content.strip().split()) #=> remove whitespaces  
    #=> above defined functions can also be used such as remove_stopwords, stem_words, lemmatize_word, and convert_number if needed
    
    return processed_content

# clean features of spaces and lowercase
def clean_content(content):
    if isinstance(content, list):
        return [i.lower().replace(' ','') for i in content]
    else:
        if isinstance(content, str):
            return content.lower().replace(' ','')
        else:
            return ''
 
  
def tfidf (corpus):
    words_set = set()
 
    for doc in  corpus:
        words = doc.split(' ')
        words_set = words_set.union(set(words))
    
    n_docs = len(corpus)         #=> number of documents in the corpus
    n_words_set = len(words_set) #=> number of unique words in the
 
    df_tf = pd.DataFrame(np.zeros((n_docs, n_words_set)), columns=words_set)
 
    # compute TF
    for i in range(n_docs):
        words = corpus[i].split(' ') #=> words in the document
        for w in words:
            df_tf[w][i] = df_tf[w][i] + (1 / len(words))
    
    # compute IDF
    idf = {}
 
    for w in words_set:
        k = 0    #=> number of documents in the corpus that contain this word
        
        for i in range(n_docs):
            if w in corpus[i].split():
                k += 1
                
        idf[w] =  np.log10(n_docs / k)
    
    # compute TF-IDF
    df_tf_idf = df_tf.copy()
 
    for w in words_set:
        for i in range(n_docs):
            df_tf_idf[w][i] = df_tf[w][i] * idf[w]
            
    return df_tf_idf 

  
# Datasets
root_path: str = "" 
#os.getcwd(encoding="utf-8")
movies_path: str = os.path.join(root_path, "datasets/tmdb_5000_movies.csv")
movies_credits_path: str = os.path.join(root_path, "datasets/tmdb_5000_credits.csv")

# Contents
#movies: str = open(movies_path, "r", encoding="utf-8", errors="ignore").read()
initial_movies = pd.read_csv(movies_path, sep=',')
initial_movies_credits = pd.read_csv(movies_path, sep=',')

#print(initial_movies)
#print(initial_movies_credits)

#
"""
Answers Section
"""
#

# shapes of dataset
print("Initial Movies:", initial_movies.shape)

print(initial_movies.head(5))

# trim dataset to include relevant features
movies_set = initial_movies[['id', 'original_title', 'genres', 'keywords', 'overview']]

print(movies_set.head(5))

# omitting any null values or redundancies
movies_set.isnull().sum()
movies_set.dropna(inplace = True)
print("Cleaned Movies:", movies_set.shape)

movies_set['genres'] = movies_set['genres'].apply(lambda x: [i['name'] for i in eval(x)])
movies_set['keywords'] = movies_set['keywords'].apply(lambda x: [i['name'] for i in eval(x)])

features = ['keywords', 'genres']
for i in features:
    movies_set[i] = movies_set[i].apply(clean_content)

print(movies_set['genres'].head(5))

movies_set['tags'] =  movies_set['keywords']  + movies_set['genres']
#movies_set['tags'] = movies_set['tags'].apply(stem)
movies_set['tags'] = movies_set['tags'].apply(lambda x: ' '.join(x))

print(movies_set['tags'].head(5))

# computing tfidf, remove stopwords and take count of 10000 most frequent words
tfidf = TfidfVectorizer(max_features=10000, stop_words ='english')

movies_set['original_title'] = movies_set['original_title'].fillna('')
movies_set['tags'] = movies_set['tags'].fillna('')

tfidf_title = tfidf.fit_transform(movies_set['original_title'])
tfidf_tags = tfidf.fit_transform(movies_set['tags'])

print(tfidf_title.shape)
print(tfidf_tags.shape)

# calcute the cosine similarity matrix
similarity_scores = cosine_similarity(tfidf_tags.toarray())

# create list of indices for later matching
indices = pd.Series(movies_set.index, index = movies_set['original_title']).drop_duplicates()


# generate top n recommendations list
def get_recommendations(title, n = 10, similarity_scores = similarity_scores):
    
    # retrieve matching movie title index
    if title not in indices.index:
        print("Movie not found.")
        return
    else:
        index = indices[title]

    # cosine similarity scores of movies in descending order
    #scores = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    similarity_scores = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)

    # get the movie indices by the scores of the n most similar movies
    similarity_scores = similarity_scores[1:n+1]
    top_n_indices = [i[0] for i in similarity_scores]
        # top n most similar movies indexes
    # use 1:n because 0 is the same movie entered
    #top_n_idx = list(scores.iloc[1:n].index)

    # Return the top 10 most similar movies
    return movies_set['original_title'].iloc[top_n_indices]

    

# results
default_request = ['Mortal Kombat', "Runaway Bride", "Scream 3"]
default_top_n = 5

print ("Question Result:")
for i in default_request :
    print(f"Top {default_top_n} recommendations for ({i}) -->\n{get_recommendations(i, default_top_n)}\n")


tfidf(movies_set['tags'])
