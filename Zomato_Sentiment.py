#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import spacy
import pandas as pd
import nltk
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#loading the spacy pretrained model on English news
nlp = spacy.load('en_core_web_md')


# In[3]:


df1 = pd.read_csv('zomato.csv')
df1.shape


# In[4]:


#print(df1)


# In[5]:


df1.head()


# In[6]:


col_list = ['name','reviews_list']
df = df1[col_list]
df.to_csv('zomatoclean.csv')
#df.shape


# In[7]:


df = pd.read_csv('zomatoclean.csv')


# In[8]:


#Just to test
df = df[:100]


# In[9]:


name = df['name']
review = df['reviews_list']


# In[10]:


#sentiment score generator function
analyser = SentimentIntensityAnalyzer()
def sentiment(s):
    score = analyser.polarity_scores(s)
    listscore = list(score.values())
    return listscore


# In[11]:


for i in range(0,review.size):
    s = str(review.loc[i])
    f = sentiment(s)
    df.loc[i, 'neg_score'] = f[0]*100
    #df.loc[i, 'neut_score'] = f[1]*100
    df.loc[i, 'pos_score'] = f[2]*100
    #df.loc[i, 'comp_score'] = f[3]*100
    
df.to_csv('zomatoscore.csv')


# In[12]:


df = pd.read_csv('zomatoscore.csv')
neg = df['neg_score']
pos = df['pos_score']


# In[13]:


print(df)


# In[15]:


#Finding max contributing word


for i in range(0,review.size):
    
    if(neg.loc[i]<pos.loc[i]):
        pword1 = ''
        pword2 = ''
        maxscore1=0.0
        maxscore2=0.0
        s = str(review.loc[i])
        wlist = s.split()
        for x in wlist:
            f = sentiment(x)
            if(maxscore1<=f[2]*100):
                maxscore1=f[2]*100
                pword1 = x
                
            if(maxscore2<=f[2]*100 and maxscore2!=maxscore1):
                maxscore2=f[2]*100
                pword2 = x
                
        
        df.loc[i,'Max_contributing_word'] = pword1
        df.loc[i,'Secondmax_contributing_word'] = pword2
        #df.loc[i,'Max_pos_score'] = maxscore
        
    elif(neg.loc[i]>=pos.loc[i]):
        nword1 = ''
        nword2 = ''
        minscore1=0.0
        minscore2=0.0
        s = str(review.loc[i])
        wlist = s.split()
        for x in wlist:
            f = sentiment(x)
            if(minscore1<=f[0]*100):
                minscore1=f[0]*100
                nword1 = x
            if(minscore2<=f[0]*100 and minscore2!=minscore1):
                minscore2=f[0]*100
                nword2 = x
        df.loc[i,'Max_contributing_word'] = nword1
        df.loc[i,'Secondmax_contributing_word'] = nword2
        #df.loc[i,'Max_neg_score'] = minscore
                

df.to_csv('zomatoword.csv')    


# In[16]:


df = pd.read_csv('zomatoword.csv')
print(df)


# In[ ]:




