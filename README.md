# Adversarial Attack on Sentiment Analysis Models

# Description

This project is a research project to test the hypothesis that models built for sentiment analysis tasks function relatively poorly when certain words are used as inputs but keeping the meaning of the input intact.

# Motivation
The motivation for this project is to examine and analyze the heavy data-driven approach of all sentiment analysis models. The corpus on which the word embeddings are trained may not have enough positive or negative connotations of certain words as a result of which the sentiment analysis models are not able to predict the sentiment with confidence.

# Explanation
Adversarial attacks are attacks on the model's performance by deliberately tweaking the original input and feeding it to the model.

Here, the original input consists of text reviews of food (Zomato) and movies (IMDB). Sentiment analysis works on identifying words in the text that have heavy positive or negative sentiment associated with them. The sequence of words matter more than the individual words and therefore an RNN-LSTM architecture is used. 
For example: The movie's plot was good. --> This is a positive feedback
             The movie's plot was not good. --> This is now a negative feedback inspite of a positive sentiment word "good" being used.
             
Our method aims to identify words (mainly adjectives and verbs) from the original reviews and replace those words with their synonyms that are farthest to them in the word embedding vector space, i.e, words having least cosine similarity. Then we use this set of modified reviews as input to the model and observe the performance.

# Results


# Conclusion
We were able to prove that our hypothesis holds true. The corpus on which word embedding algorithms are based require to have larger occurrences of rare words.
        
      
