######################################
# determine top ten LDA topics
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

moral_values = ['Generosity', 'Kindness', 'Life', 'Truth', 'Wisdom', 'Self-interest', 'Tolerance', 'Religion', 'Unity', 'Dignity', 'Courage', 'Generosity', 'Religion', 'Integrity', 'Environmentalism', 'Cooperation', 'Service', 'Compassion', 'Love', 'Justice', 'Love', 'Gratefulness', 'Communalism', 'Love', 'Obedience', 'Justice', 'Economic-Justice', 'Family', 'Peace', 'Equity', 'Respect', 'Inclusivity', 'Intellect', 'Peace', 'Understanding', 'Community', 'Individualism', 'Purity', 'Respect', 'Compassion', 'Hope', 'Repentance', 'Family', 'Non-Violence', 'Effort', 'Acceptance', 'Power', 'Fairness', 'Communalism', 'Truth', 'Peace', 'Communalism', 'Wealth', 'Harmony', 'Truth', 'Beauty', 'Openness', 'Civility', 'Compassion', 'Democracy', 'Obedience', 'Protection', 'Self-Sacrifice', 'Karma', 'Mindfulness', 'Love', 'Self-Direction', 'Peace', 'Faithfulness', 'Courage', 'Frugality', 'Meditation', 'Meditation', 'Creativity', 'Stimulation', 'Liberty', 'Humility', 'Friendship', 'Contentment', 'Selflessness', 'Attitude', 'Consequentialism', 'Hedonism', 'Humanity', 'Honor', 'Curiosity', 'Justice', 'Honesty', 'Harmony', 'Reason', 'Achievement', 'Liberty', 'Justice', 'Peace', 'Peace', 'Accommodation', 'Family', 'Science', 'Love', 'Truth', 'Justice', 'Equality', 'Straightforwardness', 'Reason', 'Contentment', 'Integrity', 'Freedom', 'Compassion', 'Science', 'Patience', 'Hospitality', 'bravery', 'Nature', 'Kindness', 'Respect', 'truthfulness', 'Acceptance', 'Goodness', 'Religion', 'Fidelity', 'Self-Control', 'Freedom', 'magnanimity', 'Descent', 'integrity', 'Reputation', 'Dignity', 'Perseverance', 'restraint', 'Love', 'politeness', 'Humanity', 'amiability', 'Humility', 'Compassion']
stop_words = set(stopwords.words('english'))
processed_values = [[word for word in value.lower().split() if word not in stop_words] for value in moral_values]

dictionary = corpora.Dictionary(processed_values)
corpus = [dictionary.doc2bow(value) for value in processed_values]

lda_train = gensim.models.ldamulticore.LdaMulticore(
   corpus,
   num_topics=10,
   id2word=dictionary,
   passes=50,
   eval_every = 1,
   per_word_topics=True)
lda_train.save('lda_train.model')
lda_topics = lda_train.print_topics()

for t in lda_topics:
    print(f"Topic {t[0]+1}: {t[1]}")
  
'''
TYPICAL RESULTS:
Topic 1: 0.092*"science" + 0.092*"harmony" + 0.048*"unity" + 0.048*"community" + 0.048*"effort" + 0.048*"bravery" + 0.048*"goodness" + 0.048*"honor" + 0.048*"restraint" + 0.048*"reputation"
Topic 2: 0.236*"love" + 0.081*"kindness" + 0.042*"descent" + 0.042*"service" + 0.042*"hedonism" + 0.042*"self-direction" + 0.042*"self-control" + 0.042*"understanding" + 0.042*"politeness" + 0.042*"self-interest"
Topic 3: 0.341*"peace" + 0.061*"faithfulness" + 0.061*"karma" + 0.061*"protection" + 0.006*"consequentialism" + 0.006*"attitude" + 0.006*"contentment" + 0.006*"hedonism" + 0.006*"honor" + 0.006*"humanity"
Topic 4: 0.130*"religion" + 0.130*"family" + 0.088*"humility" + 0.088*"liberty" + 0.046*"straightforwardness" + 0.046*"wealth" + 0.046*"hospitality" + 0.046*"purity" + 0.046*"tolerance" + 0.004*"compassion"
Topic 5: 0.156*"integrity" + 0.106*"dignity" + 0.055*"patience" + 0.055*"cooperation" + 0.055*"attitude" + 0.055*"openness" + 0.055*"beauty" + 0.055*"fidelity" + 0.005*"respect" + 0.005*"selflessness"
Topic 6: 0.111*"freedom" + 0.111*"acceptance" + 0.058*"nature" + 0.058*"truthfulness" + 0.058*"perseverance" + 0.058*"consequentialism" + 0.058*"environmentalism" + 0.058*"achievement" + 0.005*"meditation" + 0.005*"truth"
Topic 7: 0.187*"truth" + 0.096*"humanity" + 0.050*"fairness" + 0.050*"power" + 0.050*"equality" + 0.050*"economic-justice" + 0.050*"friendship" + 0.050*"honesty" + 0.050*"frugality" + 0.005*"contentment"
Topic 8: 0.171*"justice" + 0.171*"compassion" + 0.104*"respect" + 0.070*"reason" + 0.070*"obedience" + 0.037*"mindfulness" + 0.037*"gratefulness" + 0.037*"stimulation" + 0.037*"creativity" + 0.003*"liberty"
Topic 9: 0.135*"communalism" + 0.092*"contentment" + 0.048*"wisdom" + 0.048*"intellect" + 0.048*"repentance" + 0.048*"hope" + 0.048*"selflessness" + 0.048*"curiosity" + 0.048*"self-sacrifice" + 0.048*"non-violence"
Topic 10: 0.106*"generosity" + 0.106*"courage" + 0.106*"meditation" + 0.055*"democracy" + 0.055*"accommodation" + 0.055*"equity" + 0.055*"civility" + 0.055*"inclusivity" + 0.005*"humanity" + 0.005*"hedonism"
'''

######################################
# classify the Moral Values Table based on the ten LDA topics 
bow_vector = dictionary.doc2bow(moral_values)
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
  
'''
RESULTS:
Score: 0.26254600286483765   Topic: 0.096*"freedom" + 0.096*"liberty" + 0.050*"cooperation" + 0.050*"goodness" + 0.050*"perseverance"
Score: 0.13751927018165588   Topic: 0.171*"truth" + 0.130*"integrity" + 0.088*"acceptance" + 0.088*"contentment" + 0.046*"accommodation"
Score: 0.1374957263469696    Topic: 0.105*"kindness" + 0.105*"courage" + 0.055*"beauty" + 0.055*"democracy" + 0.055*"unity"
Score: 0.13748475909233093   Topic: 0.266*"love" + 0.135*"religion" + 0.048*"honor" + 0.048*"equity" + 0.048*"civility"
Score: 0.13747799396514893   Topic: 0.245*"peace" + 0.084*"meditation" + 0.044*"frugality" + 0.044*"understanding" + 0.044*"wealth"
Score: 0.13747476041316986   Topic: 0.197*"justice" + 0.081*"humility" + 0.081*"harmony" + 0.042*"nature" + 0.042*"mindfulness"
Score: 0.012500377371907234  Topic: 0.173*"family" + 0.061*"self-interest" + 0.061*"service" + 0.061*"protection" + 0.061*"stimulation"
Score: 0.012500377371907234  Topic: 0.148*"communalism" + 0.100*"dignity" + 0.053*"achievement" + 0.053*"honesty" + 0.053*"curiosity"
Score: 0.012500377371907234  Topic: 0.096*"science" + 0.096*"generosity" + 0.096*"humanity" + 0.096*"reason" + 0.050*"environmentalism"
Score: 0.01250037644058466   Topic: 0.213*"compassion" + 0.130*"respect" + 0.088*"obedience" + 0.046*"repentance" + 0.046*"fairness"
'''

######################################
# create frequency distribution graph from moral values table
import nltk
from nltk import FreqDist
moral_values = [*list of Moral Value Tables*]
dist = FreqDist(moral_values)
dist_values = dist.most_common(10)
for v in dist_values:
    print(f"{v[0]}: {v[1]}")

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







