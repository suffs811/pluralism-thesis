#!/usr/bin/python3
# author: suffs811
# Copyright (c) 2023 suffs811
# https://github.com/suffs811/pluralism-thesis

##################################
# determine the top ten LDA topics
##################################

# install necessary libraries
import os
print("### Installing necessary libraries (nltk & gensim) ###")
os.system("pip install nltk >/dev/null || echo '*** unable to install nltk; please install manually before proceeding. ***'")
os.system("pip install gensim >/dev/null || echo '*** unable to install gensim; please install manually before proceeding. ***'")

# import necessary packages
import nltk
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk import FreqDist

# set values and create list of processed words
moral_values = ['Generosity', 'Kindness', 'Life', 'Truth', 'Wisdom', 'Self-interest', 'Tolerance', 'Religion', 'Unity', 'Dignity', 'Courage', 'Generosity', 
'Religion', 'Integrity', 'Environmentalism', 'Cooperation', 'Service', 'Compassion', 'Love', 'Justice', 'Love', 'Gratefulness', 'Communalism', 'Love', 'Obedience', 
'Justice', 'Economic-Justice', 'Family', 'Peace', 'Equity', 'Respect', 'Inclusivity', 'Intellect', 'Peace', 'Understanding', 'Community', 'Individualism', 
'Purity', 'Respect', 'Compassion', 'Hope', 'Repentance', 'Family', 'Non-Violence', 'Effort', 'Acceptance', 'Power', 'Fairness', 'Communalism', 'Truth', 'Peace', 
'Communalism', 'Wealth', 'Harmony', 'Truth', 'Beauty', 'Openness', 'Civility', 'Compassion', 'Democracy', 'Obedience', 'Protection', 'Self-Sacrifice', 'Karma', 
'Mindfulness', 'Love', 'Self-Direction', 'Peace', 'Faithfulness', 'Courage', 'Frugality', 'Meditation', 'Meditation', 'Creativity', 'Stimulation', 'Liberty', 
'Humility', 'Friendship', 'Contentment', 'Selflessness', 'Attitude', 'Consequentialism', 'Hedonism', 'Humanity', 'Honor', 'Curiosity', 'Justice', 'Honesty', 
'Harmony', 'Reason', 'Achievement', 'Liberty', 'Justice', 'Peace', 'Peace', 'Accommodation', 'Family', 'Science', 'Love', 'Truth', 'Justice', 'Equality', 
'Straightforwardness', 'Reason', 'Contentment', 'Integrity', 'Freedom', 'Compassion', 'Science', 'Patience', 'Hospitality', 'bravery', 'Nature', 'Kindness', 
'Respect', 'truthfulness', 'Acceptance', 'Goodness', 'Religion', 'Fidelity', 'Self-Control', 'Freedom', 'magnanimity', 'Descent', 'integrity', 'Reputation', 
'Dignity', 'Perseverance', 'restraint', 'Love', 'politeness', 'Humanity', 'amiability', 'Humility', 'Compassion']
stop_words = set(stopwords.words('english'))
processed_values = [[word for word in value.lower().split() if word not in stop_words] for value in moral_values]

# create bag of words for lda model
dictionary = corpora.Dictionary(processed_values)
corpus = [dictionary.doc2bow(value) for value in processed_values]

# train the lda model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus, 
    num_topics=10, 
    id2word=dictionary, 
    passes=100,
    eval_every = 1)
lda_model.save('lda_train.model')
lda_topics = lda_model.print_topics()

print("\n### Top Ten LDA Topics ###\n")

# print the ten topics to screen
for t in lda_topics:
    print(f"Topic {t[0]+1}: {t[1]}")

#############################################################
# classify the Moral Values Table based on the ten LDA topics 
#############################################################

print("\n### Classification of Moral Values Based on the Ten LDA Topics ###\n")

# create bag of words from moral values list and classify topics according to values list
bow_vector = dictionary.doc2bow(moral_values)
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 1)))

#################################################################
# create frequency distribution graph from the moral values table
#################################################################

print("\n### Frequency Distribution of Top Ten Moral Values ###\n")

# create freq dist list and graph from moral values list
dist = FreqDist(moral_values)
dist_values = dist.most_common(10)
for v in dist_values:
    print(f"{v[0]}: {v[1]}")

#dist.plot(10, title="Frequency Distribution of Top Ten Moral Values")

#*** To Generate the Frequency Distribution Graph, uncomment the above statement ***
