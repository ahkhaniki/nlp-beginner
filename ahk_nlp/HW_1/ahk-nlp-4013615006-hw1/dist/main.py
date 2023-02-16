#
"""
subject: NLP HW1
autor: Amir Hussein Khaniki
"""
#

# Libraries
import os

from nltk.corpus import stopwords
from nltk import stem as stemmer, WordNetLemmatizer
from nltk import tokenize
import numpy as np
import lemm_words


# Functions

def rm_whitespace(file_path: str = "", lang="en") -> str:
    """WhiteSpace Removal in text.

    Read `txt` file and remove all white space from text and convert strings to
    lower case. main problem is punctuations.

    :param file_name
    :param lang
    """
    wst = tokenize.WhitespaceTokenizer()
    with open(file_path, "r", encoding="utf-8") as file:
        file: str = file.read()
        sk_result: list = wst.tokenize(file)
        result: str = ""
        for word in sk_result:
            result += f"{word.lower()} " if lang == "en" else f"{word}"
        return result.strip(" ")
    

c_path: str = "" 
#os.getcwd(encoding="utf-8")

# Datasets
beanstalk_text_path: str = os.path.join(c_path, "datasets/Beanstalk.txt")
short_sample_en_path: str = os.path.join(c_path, "datasets/ShortSampleEnglish.txt")
short_sample_fa_path: str = os.path.join(c_path, "datasets/ShortSamplePersian.txt")
zahak_text_path: str = os.path.join(c_path, "datasets/zahak.txt")
extra_text_path: str = os.path.join(c_path, "datasets/extra.txt")

# Contents
beanstalk_text: str = open(beanstalk_text_path, "r", encoding="utf-8", errors="ignore").read()
short_sample_en: str = open(short_sample_en_path, "r", encoding="utf-8", errors="ignore").read()
short_sample_fa: str = open(short_sample_fa_path, "r", encoding="utf-8", errors="ignore").read()
zahak_text: str = open(zahak_text_path, "r", encoding="utf-8", errors="ignore").read()
extra_text: str = open(extra_text_path, "r", encoding="utf-8", errors="ignore").read()

# print(beanstalk_text)
# print(short_sample_en)
# print(short_sample_fa)
# print(zahak_text)
# print(extra_text)



# Question 1
wst = tokenize.WhitespaceTokenizer()
tokenized_result_en = wst.tokenize(short_sample_en)
tokenized_result_fa = wst.tokenize(short_sample_fa)
tokenized_beanstalk_text = wst.tokenize(beanstalk_text)
tokenized_zahak_text = wst.tokenize(zahak_text)
tokenized_extra_text = wst.tokenize(extra_text)
print("Tokenized Result (en): ", tokenized_result_en)
print("Tokenized Result (fa): ", tokenized_result_fa)
processed_result_en = q1_result_en = " ".join(tokenized_result_en)
processed_result_fa = q1_result_fa = " ".join(tokenized_result_fa)
processed_beanstalk_text = q1_beanstalk_text = " ".join(tokenized_beanstalk_text)
processed_zahak_text = q1_zahak_text = " ".join(tokenized_zahak_text)
processed_extra_text = q1_extra_text = " ".join(tokenized_extra_text)
print("Question-1 Result (en): ", q1_result_en)
print("Question-1 Result (fa): ", q1_result_fa)

# Question 2
processed_result_en = q2_result_en = q1_result_en.lower()
processed_beanstalk_text = q2_result_beanstalk_text = q1_beanstalk_text.lower()
processed_extra_text = q2_result_extra_text = q1_extra_text.lower()
print ("Question-2 Result: ", q2_result_en)

# Question 3
pst = tokenize.PunktSentenceTokenizer()
twt = tokenize.TreebankWordTokenizer()

list_of_sentences_en = pst.tokenize(processed_result_en)
list_of_sentences_fa = pst.tokenize(processed_result_fa)
list_of_sentences_beanstalk = pst.tokenize(processed_beanstalk_text)
list_of_sentences_zahak = pst.tokenize(processed_zahak_text)
list_of_sentences_extra = pst.tokenize(processed_extra_text)

list_of_tokens_tree_en = twt.tokenize(processed_result_en)
list_of_tokens_tree_fa = twt.tokenize(processed_result_fa)
list_of_tokens_tree_beanstalk = twt.tokenize(processed_beanstalk_text)
list_of_tokens_tree_zahak = twt.tokenize(processed_zahak_text)
list_of_tokens_tree_extra = twt.tokenize(processed_extra_text)

list_of_tree_tokens_en = []
list_of_tree_tokens_fa = []
list_of_tree_tokens_beanstalk = []
list_of_tree_tokens_zahak = []
list_of_tree_tokens_extra = []

for i in list_of_sentences_en:
    list_of_tree_tokens_en.append(twt.tokenize(i))
for i in list_of_sentences_fa:
    list_of_tree_tokens_fa.append(twt.tokenize(i))
for i in list_of_sentences_beanstalk:
    list_of_tree_tokens_beanstalk.append(twt.tokenize(i))
for i in list_of_sentences_zahak:
    list_of_tree_tokens_zahak.append(twt.tokenize(i))
for i in list_of_sentences_extra:
    list_of_tree_tokens_extra.append(twt.tokenize(i))

list_of_tree_tokenz_en = []
list_of_tree_tokenz_fa = []
list_of_tree_tokenz_beanstalk = []
list_of_tree_tokenz_zahak = []
list_of_tree_tokenz_extra = []

for i in list_of_tree_tokens_en:
    for j in i:
        list_of_tree_tokenz_en.append(j)
for i in list_of_tree_tokens_fa:
    for j in i:
        list_of_tree_tokenz_fa.append(j)

for i in list_of_tree_tokens_beanstalk:
    for j in i:
        list_of_tree_tokenz_beanstalk.append(j)

for i in list_of_tree_tokens_zahak:
    for j in i:
        list_of_tree_tokenz_zahak.append(j)
for i in list_of_tree_tokens_extra:
    for j in i:
        list_of_tree_tokenz_extra.append(j)


print ("Question-3 Result:")
print ("(en)        --> Sentences = ", len(list_of_sentences_en), " , Tokens =", len(list_of_tokens_tree_en), " , Types = ", len(list(np.unique(list_of_tree_tokenz_en))), " @", list_of_sentences_en)
print ("(fa)        --> Sentences = ", len(list_of_sentences_fa), " , Tokens =", len(list_of_tokens_tree_fa), " , Types = ", len(list(np.unique(list_of_tree_tokenz_fa))),)
print ("(beanstalk) --> Sentences = ", len(list_of_sentences_beanstalk), " , Tokens =", len(list_of_tokens_tree_beanstalk), " , Types = ", len(list(np.unique(list_of_tree_tokenz_beanstalk))), )
print ("(zahak)     --> Sentences = ", len(list_of_sentences_zahak), " , Tokens =", len(list_of_tokens_tree_zahak), " , Types = ", len(list(np.unique(list_of_tree_tokenz_zahak))), )
print ("(extra)     --> Sentences = ", len(list_of_sentences_extra), " , Tokens =", len(list_of_tokens_tree_extra), " , Types = ", len(list(np.unique(list_of_tree_tokenz_extra))), )


# Question 4
regex_tokenizer = tokenize.RegexpTokenizer(r"\w+")

regexed_content_en = regex_tokenizer.tokenize(" ".join(list_of_tree_tokenz_en))
regexed_content_fa = regex_tokenizer.tokenize(" ".join(list_of_tree_tokenz_fa))
regexed_content_beanstalk = regex_tokenizer.tokenize(" ".join(list_of_tree_tokenz_beanstalk))
regexed_content_zahak = regex_tokenizer.tokenize(" ".join(list_of_tree_tokenz_zahak))
regexed_content_extra = regex_tokenizer.tokenize(" ".join(list_of_tree_tokenz_extra))

print ("Question-4 Result (punctuation-removed of en):", regexed_content_en)
print ("Question-4 Result (punctuation-removed of fa):", regexed_content_fa)
print ("Question-4 Result (punctuation-removed of beanstalk):", regexed_content_beanstalk)
print ("Question-4 Result (punctuation-removed of zahak):", regexed_content_zahak)
print ("Question-4 Result (punctuation-removed of extra):", regexed_content_extra)

# Question 5
english_stop_words = set(stopwords.words("english"))
 
filtered_content_en = []
filtered_content_beanstalk = []
filtered_content_extra = []
  
for i in regexed_content_en:
    if i not in english_stop_words:
        filtered_content_en.append(i)
               
processed_content_en = " ".join(filtered_content_en)

for i in regexed_content_beanstalk:
    if i not in english_stop_words:
        filtered_content_beanstalk.append(i)
        
processed_content_beanstalk = " ".join(filtered_content_beanstalk)

for i in regexed_content_extra:
    if i not in english_stop_words:
        filtered_content_extra.append(i)
        
processed_content_extra = " ".join(filtered_content_extra)


print ("Question-5 Result (stop-words-removed of en):", processed_content_en)


# Question 6

ps = stemmer.PorterStemmer()
lcs = stemmer.LancasterStemmer()

root_stemmer=[]
root_lancaster=[]

index_list=[2]
print ("Question-6 Result (PorterStemmer of en):")
for i in index_list:
    #root_stemmer.append(ps.stem(filtered_content_en[i]))
    print (filtered_content_en[i] , " --> " ,ps.stem(filtered_content_en[i]))

print ("Question-6 Result (LancasterStemmer of en):")
for i in index_list:
    #root_lancaster.append(lcs.stem(filtered_content_en[i]))
    print(filtered_content_en[i], " --> ", lcs.stem(filtered_content_en[i]))
    
    
index_list=[3,11,60,68]
print ("Question-6 Result (PorterStemmer of beanstalk):")
for i in index_list:
    #root_stemmer.append(ps.stem(filtered_content_beanstalk[i]))
    print (filtered_content_beanstalk[i] , " --> " ,ps.stem(filtered_content_beanstalk[i]))

print ("Question-6 Result (LancasterStemmer of beanstalk):")
for i in index_list:
    #root_lancaster.append(lcs.stem(filtered_content_beanstalk[i]))
    print(filtered_content_beanstalk[i], " --> ", lcs.stem(filtered_content_beanstalk[i]))
    
   
# Question 7


"""
Lemmatize word data table
|    words    | types |
|+-----------+|+-----+|
|     went    |   v   |
|    better   |   a   |
|     was     |   v   |
|    eaten    |   v   |
| bufferfiles |   n   |
|   fishing   |   n   |
|  signaling  |   s   |
"""

# word type, type is in range (v: verb | n: nouns | r: adverbs | a: adjective | s: satelliteAdjective)
list_of_words = [
    {
        "word": "went",
        "type": "v",
    },
    {
        "word": "better",
        "type": "a",
    },
    {
        "word": "was",
        "type": "v",
    },
    {
        "word": "eaten",
        "type": "v",
    },
    {
        "word": "bufferfiles",
        "type": "n",
    },
    {
        "word": "fishing",
        "type": "n",
    },
    {
        "word": "signaling",
        "type": "s",
    },
]


lemmatizer = WordNetLemmatizer()

root_lemmatizer = []

print ("Question-7 Result (lemmatizer without pos):")
for i in list_of_words:
    print(i, " --> ", lemmatizer.lemmatize(i["word"]))

print ("Question-7 Result (lemmatizer with pos):")
for i in list_of_words:
        print(i, " --> ", lemmatizer.lemmatize(i["word"], i["type"]))

