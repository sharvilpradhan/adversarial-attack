import nltk
from nltk.corpus import state_union

import spacy
import pandas as pd
import nltk
import pickle
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import random
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('state_union')

#loading the spacy pretrained model on English news
# nlp = spacy.load('en_core_web_md')
# df1 = pd.read_csv('zomato.csv')

# col_list = ['name','reviews_list']
# df = df1[col_list]
# df.to_csv('zomatoclean.csv')
# df = pd.read_csv('zomatoclean.csv')
# df = df[:100]
# name = df['name']
# review = df['reviews_list']

analyser = SentimentIntensityAnalyzer()
def sentiment(s):
    score = analyser.polarity_scores(s)
    listscore = list(score.values())
    return listscore

# for i in range(0,review.size):
#     s = str(review.loc[i])
#     f = sentiment(s)
#     df.loc[i, 'neg_score'] = f[0]*100
#     #df.loc[i, 'neut_score'] = f[1]*100
#     df.loc[i, 'pos_score'] = f[2]*100
#     #df.loc[i, 'comp_score'] = f[3]*100
    
# df.to_csv('zomatoscore.csv')

def POS_Tagging():
    try:
        df = pd.read_csv('processedZomato_fullFinaltest.csv')
        reviews = df['Review']
        label = df['Rating']
        word_list_col = []
        print(reviews.size)        
        for i in range(0,reviews.size):
            word_list = []
            words = nltk.word_tokenize(str(reviews.loc[i]))
            tagged = nltk.pos_tag(words)
            key_entities = list(filter(lambda word: word[1]=='JJ' or word[1]=='JJR' or word[1]=='JJS' or word[1]=='VB' or word[1]=='VBD' or word[1]=='VBG' or word[1]=='VBN' or word[1]=='VBP' or word[1]=='VBZ', tagged))
            # word_list = random.choices(key_entities, k = 12 if len(key_entities)>=12 else len(key_entities))
            for j in range(0, len(key_entities)):
                f = sentiment(key_entities[j][0])
                if label.loc[i] < 3:
                    if f[3]<0:
                        word_list.append(key_entities[j][0])
                else:
                    if f[3]>0:
                        word_list.append(key_entities[j][0])
            word_list = random.choices(word_list, k = 12) if len(word_list)>=12 else word_list

            # for j in range(0, len(word_list)):
            #     word_list[i] = word_list[i][0]
            word_list_col.append(word_list)  

        df['word_list'] = word_list_col
        
        print("generating pickle")          
        
        df.to_csv('zomato_test_reviewword.csv')

        with open("data/zomato_tagging_test.pkl", "wb") as fp:
            pickle.dump(df, fp)
            
    except Exception as e:
        print(str(e))                

# sentences = state_union.raw("2005-GWBUsh.txt").split('\n')

POS_Tagging()