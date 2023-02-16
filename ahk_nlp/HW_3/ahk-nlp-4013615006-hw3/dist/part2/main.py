#
"""
subject: NLP HW3 - Part 2
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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as sc
from sklearn.metrics import roc_auc_score

    
# Datasets
root_path: str = "" 
#os.getcwd(encoding="utf-8")
airline_tweets_path: str = os.path.join(root_path, "datasets/Tweets.csv")

# Contents
#tweety: str = open(airline_tweets_path, "r", encoding="utf-8", errors="ignore").read()
tweets = pd.read_csv(airline_tweets_path, sep=',')

#print(brown_test)
#print(brown_train)

#
"""
Answers Section
"""
#


# Question 1 and 2
data = tweets[['text', 'airline_sentiment']]

default_positive_size = data[data['airline_sentiment'] == 'positive'].size
default_positive_count = 0 #data[data['airline_sentiment'] == 'positive'].count
default_negative_size = data[data['airline_sentiment'] == 'negative'].size
default_negative_count = 0 #data[data['airline_sentiment'] == 'negative'].count
default_neutral_size = data[data['airline_sentiment'] == 'neutral'].size
default_neutral_count = 0 #data[data['airline_sentiment'] == 'neutral'].count
# drop neutral sentiment
data = data[data.airline_sentiment != "neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

# drop duplicated text
data.drop_duplicates(keep='last', inplace=True)
# drop null text
data.dropna()
tweets_list = data['text'].tolist()
cleaned_tweets_content = pd.concat([data['text'], data['airline_sentiment']], axis = 1)

print ("Question-1 Result: ")
print (f"Positive Sentiment: Size = {default_positive_size} , Count = {default_positive_count}")
print (f"Negative Sentiment: Size = {default_negative_size} , Count = {default_negative_count}")
print (f"Neutral Sentiment: Size = {default_neutral_size} , Count = {default_neutral_count}")
print (f"About Dataset:\n {data.describe()}")
print (f"Dataset Contents:\n {cleaned_tweets_content}")

word_codes, unique_words = pd.factorize(data['airline_sentiment'])
print (f"Unique Words => \n {unique_words}")
print (f"Coding => \n {word_codes}")
print(tweets_list,len(tweets_list))
data['airline_sentiment'] = pd.factorize(data['airline_sentiment'])[0]

# Question 3
max_words = 13234
max_length = 200
embed_dim = 32
lstm_out = 50

tokenizer = Tokenizer(num_words = max_words, split = ' ', lower=True)
tokenizer.fit_on_texts(data['text'].values)
sequences = tokenizer.texts_to_sequences(data['text'].values)
p_sequences = pad_sequences(sequences, maxlen = max_length, padding='post')
print (f"Sequences Model => \n {sequences}")
print (f"Pad-Sequences Model => \n {p_sequences}")

# Question 4 and 5 
x = np.array(p_sequences)
y = np.array(word_codes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

print(x.shape,y.shape)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

input_array = np.random.randint(1000, size = (32, 200))
batch_size = 32
epochs = 5

model = Sequential()
model.add(keras.layers.Embedding(13234, 32, input_length=200))
model.add(SpatialDropout1D(0.2))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='rmsprop', metrics=['accuracy'])
model.summary()
input_array = np.random.randint(1000, size=(32, 200))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape)
model.fit(x_train,y_train,epochs=5,batch_size=50)


output_array = model.predict(input_array)
print(output_array.shape)
model.fit(x_train, y_train, epochs, batch_size)

