######################################
# determine top ten LDA topics
######################################

# install necessary libraries
import os
print("### Installing necessary libraries (nltk & gensim) ###")
os.system("pip install nltk >/dev/null || echo '*** unable to install nltk; please install manually before proceeding. ***'")
os.system("pip install gensim >/dev/null || echo '*** unable to install gensim; please install manually before proceeding. ***'")

# import necessary packages
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
import gensim

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

# generate lda model
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
  
'''
TYPICAL RESULTS:
Topic 1: 0.120*"communalism" + 0.081*"courage" + 0.081*"reason" + 0.081*"acceptance" + 0.042*"honesty" + 0.042*"gratefulness" + 0.042*"selflessness" + 
0.042*"self-direction" + 0.042*"understanding" + 0.042*"cooperation"
Topic 2: 0.096*"dignity" + 0.096*"science" + 0.096*"kindness" + 0.096*"obedience" + 0.050*"friendship" + 0.050*"hedonism" + 0.050*"curiosity" + 0.050*"mindfulness" 
+ 0.050*"faithfulness" + 0.005*"justice"
Topic 3: 0.156*"integrity" + 0.106*"contentment" + 0.055*"self-interest" + 0.055*"hospitality" + 0.055*"purity" + 0.055*"truthfulness" + 0.055*"democracy" + 
0.055*"civility" + 0.005*"obedience" + 0.005*"consequentialism"
Topic 4: 0.142*"religion" + 0.096*"humility" + 0.050*"economic-justice" + 0.050*"patience" + 0.050*"creativity" + 0.050*"self-control" + 0.050*"community" + 
0.050*"politeness" + 0.050*"descent" + 0.050*"reputation"
Topic 5: 0.061*"goodness" + 0.061*"hope" + 0.061*"protection" + 0.061*"amiability" + 0.061*"intellect" + 0.061*"openness" + 0.061*"equity" + 0.061*"karma" + 
0.061*"fidelity" + 0.006*"attitude"
Topic 6: 0.205*"compassion" + 0.084*"humanity" + 0.084*"liberty" + 0.084*"harmony" + 0.044*"attitude" + 0.044*"equality" + 0.044*"accommodation" + 0.044*"wisdom" + 
0.044*"life" + 0.004*"hedonism"
Topic 7: 0.179*"truth" + 0.135*"respect" + 0.092*"generosity" + 0.048*"wealth" + 0.048*"self-sacrifice" + 0.048*"restraint" + 0.048*"honor" + 0.048*"fairness" + 
0.004*"communalism" + 0.004*"justice"
Topic 8: 0.106*"meditation" + 0.106*"freedom" + 0.055*"power" + 0.055*"environmentalism" + 0.055*"achievement" + 0.055*"straightforwardness" + 0.055*"perseverance" 
+ 0.055*"frugality" + 0.055*"bravery" + 0.005*"respect"
Topic 9: 0.255*"love" + 0.130*"family" + 0.046*"unity" + 0.046*"stimulation" + 0.046*"beauty" + 0.046*"repentance" + 0.046*"service" + 0.046*"effort" + 
0.004*"reason" + 0.004*"curiosity"
Topic 10: 0.245*"peace" + 0.205*"justice" + 0.044*"nature" + 0.044*"individualism" + 0.044*"non-violence" + 0.044*"consequentialism" + 0.044*"inclusivity" + 
0.004*"integrity" + 0.004*"communalism" + 0.004*"harmony"
'''

######################################
# classify the Moral Values Table based on the ten LDA topics 
######################################

print("\n### Classification of Moral Values Based on the Ten LDA Topics ###\n")

# create bag of words from moral values list and classify topics according to values list
bow_vector = dictionary.doc2bow(moral_values)
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 1)))

'''
TYPICAL RESULTS:
One Topic Word:
Score: 0.26256901025772095   Topic: 0.156*"integrity"
Score: 0.1375010907649994    Topic: 0.061*"equity"
Score: 0.13749247789382935   Topic: 0.106*"meditation"
Score: 0.13748463988304138   Topic: 0.142*"religion"
Score: 0.13748086988925934   Topic: 0.179*"truth"
Score: 0.13747039437294006   Topic: 0.120*"communalism"
Score: 0.012500380165874958  Topic: 0.096*"kindness"
Score: 0.012500379234552383  Topic: 0.205*"compassion"
Score: 0.012500379234552383  Topic: 0.255*"love"
Score: 0.012500379234552383  Topic: 0.245*"peace"
'''

######################################
# create frequency distribution graph from moral values table
######################################

print("\n### Frequency Distribution of Top Ten Moral Values ###\n")

# create freq dist list and graph from moral values list
dist = FreqDist(moral_values)
dist_values = dist.most_common(10)
for v in dist_values:
    print(f"{v[0]}: {v[1]}")

#dist.plot(10, title="Frequency Distribution of Top Ten Moral Values")

#*** To Generate the Frequency Distribution Graph, uncomment the above statement ***

'''
RESULTS:

Love: 6
Peace: 6
Compassion: 5
Justice: 5
Truth: 4
Religion: 3
Communalism: 3
Family: 3
Respect: 3
Generosity: 2
'''
