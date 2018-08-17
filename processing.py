""" Twitter Sentiment Analysis
#Supervised Classification
#The twitter_samples corpus contains 3 files.
1) negative_tweets.json: contains 5k negative tweets
2) positive_tweets.json: contains 5k positive tweets
3) tweets.20150430-223406.json: contains 20k positive and negative tweets"""
from nltk.corpus import twitter_samples
print(twitter_samples.fileids())

pos_tweets = twitter_samples.strings('positive_tweets.json')
print(len(pos_tweets))
neg_tweets = twitter_samples.strings('negative_tweets.json')
print(len(neg_tweets))
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')

print(len(all_tweets))
for tweet in pos_tweets[:5]:
  print(tweet)

"""Tokenize Tweets
NLTK has a TweetTokenizer module that does a good job in tokenizing (splitting text into a list of words) tweets."""
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
for tweet in pos_tweets[:5]:
    print(tweet_tokenizer.tokenize(tweet))


"""Cleaning Tweet
In the tweet cleaning process, we will:
– Remove stock market tickers like $GE
– Remove retweet text “RT”
– Remove hyperlinks
– Remove hashtags ( only the hashtag # and not the word)
– Remove stop words like a, and, the, is, are, etc.
– Remove emoticons like :), :D, :(, :-), etc.
– Remove punctuation like full-stop, comm a, exclamation sign, etc.
– Convert words to Stem/Base words using Porter Stemming Algorithm.
We will define a function named clean_tweets which returns a list of cleaned (by removing the
above-mentioned things) words for any given tweet."""
import string
import re
from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.tokenize import TweetTokenizer
"""# Happy Emoticons
emoticons_happy = set({':-)',':)',';)',':o)',':]',':3',':c)',':>','=]','8)','=)',':}',':^)',
':-D',':D', '8-D','8D','x-D', 'xD','X-D','XD','=-D','=D','=-3','=3',':-))',':*',':^*','>:P',
':-P',':P','XP','x-p','xp','XP',':-p',':p','=p',':-b',':b','>:)','>;)','>:-)','<3'})
# Sad Emoticons
emoticons_sad = set([':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',':c', ':{', '>:\\', ';('])
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)"""

def clean_tweets(tweet):

# remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

# remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

# remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

# remove hashtags
# only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

# tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if(word not in stopwords_english and # remove stopwords
             #word not in emoticons and # remove emoticons
              word not in string.punctuation): # remove punctuation
#tweets_clean.append(word)
           stem_word = stemmer.stem(word) # stemming word
           tweets_clean.append(stem_word)
    return tweets_clean
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
# print cleaned tweet
print(clean_tweets(custom_tweet))
print(pos_tweets[5])
print(clean_tweets(pos_tweets[5]))


#Feature Extraction
#We define a simple bag_of_words(BOW) function that extracts unigram features from the tweets.
# feature extractor function
def bag_of_words(tweet):
    words = clean_tweets(tweet)
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
print (bag_of_words(custom_tweet))


# positive tweets feature set
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))


# negative tweets feature set
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))
print (len(pos_tweets_set), len(neg_tweets_set)) # Output: (5000, 5000)


"""Create Train and Test Set
There are 5000 positive tweets set and 5000 negative tweets set. We take 20% (i.e. 1000) of positive
tweets and 20% (i.e. 1000) of negative tweets as the test set. The remaining negative and positive
tweets will be taken as the training set."""
# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
from random import shuffle
shuffle(pos_tweets_set)
shuffle(neg_tweets_set)
test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
print(len(test_set),len(train_set)) # Output: (2000, 8000)

"""Training Classifier and Calculating Accuracy
We train Naive Bayes Classifier using the training set and calculate the classification accuracy of the
trained classifier using the test set."""
from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_set)
accuracy = classify.accuracy(classifier, test_set)
print(accuracy) # Output: 0.765
print (classifier.show_most_informative_features(10))

"""Testing Classifier with Custom Tweet
We provide custom tweet and check the classification output of the trained classifier. The classifier
correctly predicts both negative and positive tweets provided."""
custom_tweet = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_tweet_set = bag_of_words(custom_tweet)
print (classifier.classify(custom_tweet_set)) # Output: neg
# Negative tweet correctly classified as negative
# probability result
prob_result = classifier.prob_classify(custom_tweet_set)
print (prob_result) # Output: <ProbDist with 2 samples>
print (prob_result.max()) # Output: neg
print (prob_result.prob("neg")) # Output: 0.941844352481
print (prob_result.prob("pos")) # Output: 0.0581556475194
custom_tweet = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
custom_tweet_set = bag_of_words(custom_tweet)
print (classifier.classify(custom_tweet_set)) # Output: pos
# Positive tweet correctly classified as positive
# probability result
prob_result = classifier.prob_classify(custom_tweet_set)
print (prob_result) # Output: <ProbDist with 2 samples>
print (prob_result.max()) # Output: pos
print (prob_result.prob("neg")) # Output: 0.00131055449755
print (prob_result.prob("pos")) # Output: 0.998689445502

"""Precision, Recall & F1-Score"""
from collections import defaultdict
actual_set = defaultdict(set)
predicted_set = defaultdict(set)
actual_set_cm = []
predicted_set_cm = []
for index, (feature, actual_label) in enumerate(test_set):
    actual_set[actual_label].add(index)
    actual_set_cm.append(actual_label)
    predicted_label = classifier.classify(feature)
    predicted_set[predicted_label].add(index)
    predicted_set_cm.append(predicted_label)
from nltk.metrics import precision, recall, f_measure, ConfusionMatrix
print('pos precision:', precision(actual_set['pos'], predicted_set['pos'])) #Output: pos precision: 0.762896825397
print('pos recall:', recall(actual_set['pos'], predicted_set['pos'])) # Output: pos recall: 0.769
print('pos F-measure:', f_measure(actual_set['pos'], predicted_set['pos'])) #Output: pos F-measure: 0.76593625498
print('neg precision:', precision(actual_set['neg'], predicted_set['neg'])) #Output: neg precision: 0.767137096774
print('neg recall:', recall(actual_set['neg'], predicted_set['neg'])) # Output: neg recall: 0.761
print('neg F-measure:', f_measure(actual_set['neg'], predicted_set['neg']))#Output: neg F-measure: 0.7640562249


"""Confusion Matrix"""
cm = ConfusionMatrix(actual_set_cm, predicted_set_cm)
print (cm)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))



