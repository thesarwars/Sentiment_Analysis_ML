import joblib
# import numpy as np
# import pandas as pd
import nltk
nltk.data.path.append('/nltk_data')
import pickle
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import re
import string
import time
from matplotlib import pyplot
# import streamlit as st
import json



# @st.cache_data
# def load_model(vectoriser_path, model_path):
    
#     # Load the vectoriser.
#     file = open(vectoriser_path, 'rb')
#     vectoriser = pickle.load(file)
#     file.close()
    
#     # Load the LR Model.
#     file = open(model_path, 'rb')
#     LRmodel = pickle.load(file)
#     file.close()
#     print('Model Loaded Successfully')
#     return vectoriser, LRmodel

def load_model(vectoriser_path, model_path):
    
    # Load the vectoriser using joblib
    vectoriser = joblib.load(vectoriser_path)

    # Load the LR model using joblib
    LRmodel = joblib.load(model_path)

    print('Model Loaded Successfully')
    return vectoriser, LRmodel

def inference(vectoriser, model, tweets, cols):
    text = tweets.split('; ')
    finaldata = []
    textdata = vectoriser.transform(lemmatize_process(preprocess(text)))
    # textdata = vectoriser.transform(lemmatize_process(preprocess(text)))
    sentiment = model.predict(textdata)
    
    # print(model.classes_)
    sentiment_prob = model.predict_proba(textdata)
    
    for index,tweet in enumerate(text):
        if sentiment[index] == 1:
            sentiment_probFinal = sentiment_prob[index][1]
        else:
            sentiment_probFinal = sentiment_prob[index][0]
            
        sentiment_probFinal2 = "{}%".format(round(sentiment_probFinal*100,2))
        finaldata.append((tweet, int(sentiment[index]), (sentiment_probFinal2)))
           
    # Convert the list into a Pandas DataFrame.
    # df = pd.DataFrame(finaldata, columns = ['Tweet','Sentiment', 'Probability(Confidence Level)'])
    # df = df.replace([0,1], ["Negative","Positive"])
    # jsonDf = json.dumps(finaldata)
    
    return finaldata
    
def get_wordnet_pos_tag(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN    
    
    

def lemmatize_process(preprocessedtext):
    """ the process of reducing the different forms of a word to one single form,
    we would lemmatize the text we provid here.
    """
    
    lemma = WordNetLemmatizer()
    
    finalprocessedtext = []
    for tweet in preprocessedtext:
        text_pos = pos_tag(word_tokenize(tweet))
        words = [x[0] for x in text_pos]
        pos = [x[1] for x in text_pos]
        tweet_lemma = " ".join([lemma.lemmatize(a,get_wordnet_pos_tag(b)) for a,b in zip(words,pos)])
        finalprocessedtext.append(tweet_lemma)
    return finalprocessedtext


    
def preprocess(textdata):
    """ preprocess the tweet or a post by this 'preprocess' method. Where
     text would filter by this emoji, stopwords, regex and punctuation.
    """
    
    # Defining dictionary containing all emojis with their meanings.
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                 'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                 'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                 'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
                 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                 'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                 'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                 'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                 's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                 't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
                 'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                 'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                 'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                 "youve", 'your', 'yours', 'yourself', 'yourselves']
    
    processedText = []
    
    # wordLem = WordNetLemmatizer()
    # Define regex patter for filter text
    # url_pattern = r"((http://)[^ ]*(https://)[^ ]*(www.\)[^ ]*)"
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    char_pattern = "^[a-zA-z]"
    sequencePattern = r"(.)1\1\+"
    replaceSeqPatt = r"1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(url_pattern, ' URL', tweet)
        
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, 'EMOJI' + emojis[emoji])
        
        # Replace @USERNAME to 'USER' if any.  
        tweet = re.sub(user_pattern, 'USER', tweet)
            
        # Replace all non alphabets.
        tweet = re.sub(char_pattern, '', tweet)
        
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, replaceSeqPatt, tweet)
        
        #Removing punctuations if any left post removing all all non alphabets
        all_char_list = []
        all_char_list = [char for char in tweet if char not in string.punctuation]
        tweet = ''.join(all_char_list)
        
        # Removing all stopwords as per custom list defined above
        tweetwords = ''
        for word in tweet.split():
            if word not in (stopwordlist):
                if len(word)>1:
                    tweetwords += (word+' ')
                    
        processedText.append(tweetwords)
    return processedText