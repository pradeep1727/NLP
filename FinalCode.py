#!/usr/bin/env python
# coding: utf-8




import re
import nltk
import pickle
import string
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import wordnet
#from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier




data = pd.read_csv('data.csv')
data.head()


# # Extract Tags and Complaint columns from dataset




df = data[['Tags','Complaint']]
df['Tags'] = df['Tags'].str.lower()
df.head()


# # Stopwords




add_stop_words= ['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',              'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',             'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',              'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',             'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',              's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',              've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',             "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",              'won', "won't", 'wouldn', "wouldn't"]

#re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

#add words that aren't in the NLTK stopwords list
new_stopwords_list = stop_words.union(add_stop_words)

#remove words that are in NLTK stopwords list
not_stopwords = {'no','not'} 
final_stop_words = set([word for word in new_stopwords_list if word not in not_stopwords])

re_stop_words = re.compile(r"\b(" + "|".join(final_stop_words) + ")\\W", re.I)


# # Data Ceaning




def decontract(sentence):
    # specific
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

def cleanPunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub('[%s]' % re.escape(string.punctuation), '', cleaned) #removing punctuation
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = re.sub(r"\s+[a-zA-Z]\s+", " ", cleaned)#removing single character
    cleaned = re.sub('\w*\d\w*', ' ', cleaned) #removing digits
    cleaned = re.sub(r'\bx\w*', ' ', cleaned)  #removing words beginning with x
    cleaned = re.sub('\n', '', cleaned) #removing new line character
    cleaned = re.sub(r"\s+"," ", cleaned) #removing extra spaces
    cleaned = re.sub(r"^\s+", "", cleaned)#removing extra spaces from start
    cleaned = re.sub(r"\s+$", "", cleaned)#removing extra spaces from end
    cleaned = re.sub(r"\s+[a-zA-Z]\s+", " ", cleaned)#removing single character
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_word = re.sub(r'\bx\w*','', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub("", sentence)







# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None


lemmatizer = WordNetLemmatizer()


def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)







def checked(text):
    word_tokens = word_tokenize(text)
    checked_list = []
    spell = SpellChecker()
    misspelled = spell.unknown(word_tokens)
    for w in word_tokens:
        if w not in misspelled: 
            checked_list.append(w)
       
    return " ".join(checked_list)








# # Cleaning Complaint column




df['Complaint'] = df['Complaint'].str.lower()
df['Complaint'] = df['Complaint'].apply(decontract)
df['Complaint'] = df['Complaint'].apply(cleanPunc)
df['Complaint'] = df['Complaint'].apply(keepAlpha)
df['Complaint'] = df['Complaint'].apply(removeStopWords)
df['Complaint'] = df['Complaint'].apply(lemmatize_sentence)
df['Complaint'] = df['Complaint'].apply(checked)





df['Tags'] = df['Tags'].str.lower()
df['Tags'] = df['Tags'].apply(cleanPunc)
df['Tags'] = df['Tags'].apply(checked)







word_tokens = []
for sent in df['Tags']:
    x = word_tokenize(sent)
    word_tokens.append(x)
df['Tags'] = word_tokens
df.head()


# # Prepare Y variable



multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['Tags'])

# transform target variable
y = multilabel_binarizer.transform(df['Tags'])




#xtrain, xval, ytrain, yval = train_test_split(df['Complaint'], y, test_size=0.2, random_state=9)

xtrain = df['Complaint']
ytrain = y



cv = TfidfVectorizer(max_df=0.7, max_features=10000)
# create TF-IDF features
xtrain_tfidf = cv.fit_transform(xtrain)
#xval_tfidf = cv.transform(xval)





#xg = XGBClassifier()
xg = RandomForestClassifier()
clf = OneVsRestClassifier(xg)
# fit model on train data
clf.fit(xtrain_tfidf, ytrain)




#y_pred = clf.predict(xval_tfidf)


def prediction(q):
    q = str.lower(q)
    q = decontract(q)
    q = cleanPunc(q)
    q = keepAlpha(q)
    q = removeStopWords(q)
    q = lemmatize_sentence(q)
    q = checked(q)
    q_vec = cv.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)