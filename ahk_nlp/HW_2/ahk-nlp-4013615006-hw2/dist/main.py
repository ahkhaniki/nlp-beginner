#
"""
subject: NLP HW2
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
from collections import Counter as ngram_counter
from regex import match as re_match
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams, bigrams, trigrams, stem as stemmer, WordNetLemmatizer as lemmatizer, lm, FreqDist, LaplaceProbDist, SimpleGoodTuringProbDist
# import inflect
# p = inflect.engine()

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
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems

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

# extract n-grams from text/content
def ngram(content, n):
    tokens = word_tokenize(content) #=> tokenize text to the words
    ngram_result = list(ngrams(tokens, n)) #=> extraxt n-grams from tokens based on n, for example n = 1 as unigram or n = 2 as bigram
    return ngram_result

# count n-grams repeat 
def count_ngrams(ngram):
    return ngram_counter(ngram)

# take top n of n-gram set
def take_top(ngram_set, n):
    return ngram_set.most_common(n)

# smoothing n-gram set
def smooth(ngram, n):
    freqDist = FreqDist(ngram)
    if n == 1:
        laplaceDist = LaplaceProbDist(freqDist, bins = freqDist.N())
        return laplaceDist
    else:
        turingDist = SimpleGoodTuringProbDist(freqDist, bins = freqDist.N())
        return turingDist

# calculates the frequency of the (i+1)th word in the whole corpus/set   
def next_word_freq(array, sentence):
    sen_len, word_list = len(sentence.split()), []
        
    for i in range(len(array)):
        if ' '.join(array[i : i + sen_len]).lower() == sentence.lower():
            if i + sen_len < len(array) - 1:
                word_list.append(array[i + sen_len])
    
    return dict(ngram_counter(word_list)) #=> count of each word in word_list

# calculate the CDF of each word
def cdf(d):
    prob_sum, sum_vals = 0, sum(d.values())
    for k, v in d.items():
        pmf = v / sum_vals
        prob_sum += pmf
        d[k] = prob_sum
    
    return d

# predict next word sentence/word by n-word
def predict_words(entered_phrase, phrase_len, words_number, corpus = ''):
    corpus = default_corpus if corpus == '' else corpus
    l = corpus.split()
    temp_out = ''
    out = entered_phrase + ' '
    for i in range(words_number - phrase_len):
        func_out = next_word_freq(l, entered_phrase)
        cdf_dict = cdf(func_out)
        rand = random.uniform(0, 1)
        try: key, val = zip(*cdf_dict.items())
        except: break
        for j in range(len(val)):
            if rand <= val[j]:
                pos = j
                break
        temp_out = key[pos]
        out = out + temp_out + ' '
        entered_phrase = temp_out
    return out

# finding perplexity for test data
# data source: https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954
def find_perplexity(self, text, n, vocab, count, context):
    """
    This method takes the test data, first generate the n-gram model of text, then find the preplexity of test data on our trained model.
    Parameters
    text: String
        The test text, in which we we want to test our model
    n: int
        n refers to the model we want to create, example: unigram, bigram....
    vocab: list
        list containing unique vocabulary (list of characters)
    count: Dict
        it contains the count of each value (n-gram) model .
    context: Counter object (dict) if n>1 | int if n==1
        the count on the context values.
    Returns
    perplexity: float
        returns perplexity of our model
    """
    # obtaining the tokens from the text
    t_tokens = []
    for line in text:
        word = line.split()
        for w in word:
            for c in w:
                t_tokens.append(c)
    # appending the "<UNK> token in case my vocab does not have this token"
    new_t_tokens = []
    for tt in t_tokens:
        if tt in vocab:
            new_t_tokens.append(tt)
        else:
            new_t_tokens.append("<UNK>")
    # finding the probability of the model
    probability, log_probability = 0, 0
    # to find the denominator
    model = ngrams(new_t_tokens, n)
    for item in model:
        # unigram model (the context is N: length of token)
        if n == 1:
            log_probability = np.log2(count[item] / context)
        # for n-gram model where n>=2
        else:
            if (context[item[:-1]] == 0):
                log_probability = 0
            else:
                log_probability = np.log2(count[item] / context[item[:-1]])

        probability = probability + log_probability
    perplexity = np.power(2, (-1 / len(new_t_tokens)) * probability)
    return perplexity

    
# Datasets
root_path: str = "" 
#os.getcwd(encoding="utf-8")
brown_test_path: str = os.path.join(root_path, "datasets/brown.test.txt")
brown_train_path: str = os.path.join(root_path, "datasets/brown.train.txt")

# Contents
brown_test: str = open(brown_test_path, "r", encoding="utf-8", errors="ignore").read()
brown_train: str = open(brown_train_path, "r", encoding="utf-8", errors="ignore").read()

#print(brown_test)
#print(brown_train)

#
"""
Answers Section
"""
#

# Question 1
preprocessed_content = pre_process(brown_test)
default_corpus = preprocessed_content
print ("Question-1 Result: ", preprocessed_content)

# Question 2
tokens = word_tokenize(preprocessed_content)

#unigram = list(ngrams(tokens, 1))
#unigram_count = ngram_counter(unigram)
#unigram_top = unigram_count.most_common(5)
unigram = ngram(preprocessed_content, 1)
unigram_count = count_ngrams(unigram)
unigram_top = take_top(unigram_count, 5)

#bigram = list(ngrams(tokens, 2)) #=> list(bigrams(tokens))
#bigram_count = ngram_counter(bigram)
#bigram_top = bigram_count.most_common(5)
bigram = ngram(preprocessed_content, 2)
bigram_count = count_ngrams(bigram)
bigram_top = take_top(bigram_count, 5)

#trigram = list(ngrams(tokens, 3)) #=> list(trigrams(tokens))
#trigram_count = ngram_counter(trigram)
#trigram_top = trigram_count.most_common(5)
trigram = ngram(preprocessed_content, 3)
trigram_count = count_ngrams(trigram)
trigram_top = take_top(trigram_count, 5)

#quadgram = list(ngrams(tokens, 4))
#quadgram_count = ngram_counter(quadgram)
#quadgram_top = quadgram_count.most_common(5)
quadgram = ngram(preprocessed_content, 4)
quadgram_count = count_ngrams(quadgram)
quadgram_top = take_top(quadgram_count, 5)

print ("Question-2 Result: ")
print(f"Unigram -> Count = {len(unigram)},  Top = {unigram_top}")
print(f"Bigram -> Count = {len(bigram)},  Top = {bigram_top}")
print(f"Trigram -> Count = {len(trigram)},  Top = {trigram_top}")
print(f"Quadgram -> Count = {len(quadgram)},  Top = {quadgram_top}")

# Question 3
print ("Question-3 Result: ")
print(f"Unigram -> smoothing:: {smooth(unigram, 1)}")
print(f"Bigram -> smoothing:: {smooth(bigram, 2)}")
print(f"Trigram -> smoothing:: {smooth(trigram, 3)}")
print(f"Quadgram -> smoothing:: {smooth(quadgram, 4)}")


# Question 4 and 5 
sample_input = 'why'
predict_words(sample_input, len(sample_input), 20)


# Question 6
train_sentences = ['the best','assume']
tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in train_sentences]
n = 1
train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
model = lm.MLE(n)
model.fit(train_data, padded_vocab)

test_sentences = ['the best', 'between']
tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in test_sentences]

test_data, _ = padded_everygram_pipeline(n, tokenized_text)
for test in test_data:
    print ("MLE estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

test_data, _ = padded_everygram_pipeline(n, tokenized_text)

for i, test in enumerate(test_data):
  print(f"Perplexity ({test_sentences[i]}): {model.perplexity(test)}")