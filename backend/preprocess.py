# bootstrap nlp libraries
import re
import demoji
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
wl = WordNetLemmatizer()
demoji.download_codes()

# contractions regex
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }
contracts = re.compile(r'\b(' + '|'.join(contraction_mapping.keys()) + r')\b')

# extract textual features in-place (for external comparative datasets)
def feature_extract(df, tweet):
    df['emoji_count'] = df[tweet].apply(lambda t: len(demoji.findall_list(t)))
    df['polarity'] = df[tweet].apply(lambda t: TextBlob(t).polarity)
    df['subjectivity'] = df[tweet].apply(lambda t: TextBlob(t).subjectivity)
    df['hashtag_count'] = df[tweet].str.count('#')
    df['mentions_count'] = df[tweet].str.count('@')
    df['words_count'] = df[tweet].str.split().str.len()
    df['char_count'] = df[tweet].str.len()
    df['url_count'] = df[tweet].str.count('https?://\S+')
    df['is_retweet'] = df[tweet].apply(lambda t: 1 if re.search('[Rr][Tt].@\S+', t) else 0)

# tweet normalisation/cleaning
def clean(tweet):
    # drop all urls
    no_urls = re.sub(r'(((https?:\s?)?(\\\s*/\s*)*\s*t\.co\s*(s*\\\s*/\s*)*\S+)|https?://\S+)', '', tweet)
    # transform emojis to their description
    new = demoji.replace_with_desc(string=no_urls, sep='\"')
    # fix html encoded chars
    new = BeautifulSoup(new, 'lxml').get_text()
    # remove retweet tags
    no_retweets = re.sub("[Rr][Tt].@\S+", " ", new)
    # lower case all words
    lower_case = no_retweets.lower()
    # fix negated words with apostrophe
    negations_fix = contracts.sub(lambda x: contraction_mapping[x.group()], lower_case)
    # remove all mentions
    no_mentions = re.sub("@\S+", " ", negations_fix)
    # remove all special characters
    letters_only = re.sub("[^a-zA-Z]", " ", no_mentions)
    # tokenize
    words = [x for x in nltk.word_tokenize(letters_only) if len(x) > 1]
    # remove stop words
    no_stopwords = list(filter(lambda l: l not in stop_words, words))
    # lemmatize words
    lemmas = [wl.lemmatize(t) for t in no_stopwords]
    return (" ".join(lemmas)).strip()

def process(df, tweet):
    df[f'{tweet}_clean'] = df[tweet].apply(clean)