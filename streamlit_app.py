import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# imports to allow pip installs
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("successfully installed", package)

# imports other than streamlit
import pandas as pd
import math
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# import matplotlib
import numpy as np
install("plotly")
import plotly.figure_factory as ff

import seaborn as sns

from nltk.corpus import stopwords
from collections import  Counter

import sklearn
# import gensim

import pyLDAvis
import wordcloud
import textblob
import spacy
import textstat

from textstat import flesch_reading_ease

import nltk
nltk.download('stopwords')
stop=set(stopwords.words('english'))
nltk.download('punkt')

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# GET HEADLINE DATA
def get_news_headline_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/abcnews-date-text.csv'
    return pd.read_csv(DATA_FILENAME)

news = get_news_headline_data()


def st_plot(use_cont_width = True):

    fig = plt.gcf()
    st.plotly_chart(fig, use_container_width = use_cont_width)

# Plots headline lengths
text = news['headline_text']
# text.str.len().hist()

text.str.len().hist()

st_plot()

# Code Snippet for Top Stopwords Barchart
from nltk.corpus import stopwords

def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1

    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
    x,y=zip(*top)
    plt.bar(x,y)

plot_top_stopwords_barchart(text)

st_plot()

# Create a histogram
# plt.hist(data, bins=20, color='skyblue', edgecolor='black')

# # Add title and labels
# plt.title('Interactive Histogram with Streamlit')
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')

# fig = plt.gcf()

# # Display the histogram
# # plt.show()
# st.plotly_chart(fig, use_container_width = False)

# * TOPIC MODELING *
# NOTE: try using other models other than pyLDAvis

print("here")
# * preprocessing*

# * Topic *

# * WORDCLOUD *
# Create some sample text
text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot()

# reading = news['headline_text'].apply(lambda x : flesch_reading_ease(x))
# reading.hist()

# * Sentiment Analysis *
# NOTE: I tried textblob and vader models
# try using other models as well

# Method 1: textblob
from textblob import TextBlob

# example
# TextBlob('100 people killed in Iraq').sentiment

def st_plot_polarity_histogram(text):

    def _polarity(text):
        return TextBlob(text).sentiment.polarity

    news['polarity_score'] = text.apply(lambda x : _polarity(x))
    news['polarity_score'].hist()

    st_plot()

_='''
st_plot_polarity_histogram(news['headline_text'])
'''

def sentiment(x):
    if x<0:
        return 'neg'
    elif x==0:
        return 'neu'
    else:
        return 'pos'

_='''
news['polarity'] = news['polarity_score'].map(lambda x: sentiment(x))

plt.bar(news.polarity.value_counts().index, 
        news.polarity.value_counts())

st_plot()

'''

# st.write(
#     "a few of the positive headlines"
# )
# news[news['polarity']=='pos']['headline_text'].head()

# st.write(
#     "a few of the negative headlines"
# )
# news[news['polarity']=='neg']['headline_text'].head()

# download_thread = threading.Thread(target=get_lda_objects, name="LDA", args=news['headline_text'])
# download_thread.start()

# Method 2: Vader
# Code Snippet for Sentiment Barchart

from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def sentiment_vader(text, sid):
    ss = sid.polarity_scores(text)
    ss.pop('compound')
    return max(ss, key = ss.get)

def sentiment_textblob(text):
        x = TextBlob(text).sentiment.polarity

        if x<0:
            return 'neg'
        elif x==0:
            return 'neu'
        else:
            return 'pos'

def plot_sentiment_barchart(text, method='TextBlob'):
    if method == 'TextBlob':
        sentiment = text.map(lambda x: sentiment_textblob(x))
    elif method == 'Vader':
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        sentiment = text.map(lambda x: sentiment_vader(x, sid=sid))
    else:
        raise ValueError('Textblob or Vader')

    plt.bar(sentiment.value_counts().index,
            sentiment.value_counts())
    
    st_plot()

# download_thread = threading.Thread(target=plot_sentiment_barchart, name="Plot_TextBlob", args=news['headline_text'], method='TextBlob')
# download_thread.start()
_='''
plot_sentiment_barchart(news['headline_text'], method='TextBlob')
plot_sentiment_barchart(news['headline_text'], method='Vader')
'''

# * NAMED ENTITY RECOGNITION *
import spacy

# subprocess.check_call([sys.executable, "-m", "python", "spacy download", "en_core_web_lg"])
# subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"]) 
print("successfully installed", "package")

nlp = spacy.load("en_core_web_sm")

doc = nlp('India and Iran have agreed to boost the economic viability of the strategic Chabahar port through various measures, including larger subsidies to merchant shipping firms using the facility, people familiar with the development said on Thursday.')

[(x.text,x.label_) for x in doc.ents]

print("1")
from spacy import displacy

print("2")
displacy.render(doc, style='ent')

print("3")
def ner(text):
    doc=nlp(text)
    print("4.2")
    return [X.label_ for X in doc.ents]

print("4.1")
ent=news['headline_text'].apply(lambda x : ner(x))
print("4.5")
ent=[x for sub in ent for x in sub]

print("5")
counter=Counter(ent)
count=counter.most_common()

print("hi")
# Code Snippet for Named Entity Barchart

import spacy
from collections import  Counter
import seaborn as sns

def plot_named_entity_barchart(text):
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text):
        doc = nlp(text)
        return [X.label_ for X in doc.ents]

    ent=text.apply(lambda x : _get_ner(x))
    ent=[x for sub in ent for x in sub]
    counter=Counter(ent)
    count=counter.most_common()

    x,y=map(list,zip(*count))
    sns.barplot(x=y,y=x)

plot_named_entity_barchart(news['headline_text'])

import spacy
from collections import  Counter
import seaborn as sns

def plot_most_common_named_entity_barchart(text, entity="PERSON"):
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text,ent):
        doc=nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]

    entity_filtered=text.apply(lambda x: _get_ner(x,entity))
    entity_filtered=[i for x in entity_filtered for i in x]

    counter=Counter(entity_filtered)
    x,y=map(list,zip(*counter.most_common(10)))
    sns.barplot(x = y, y = x).set_title(entity)

plot_most_common_named_entity_barchart(news['headline_text'], entity="PERSON")


# Code Snippet for Most Common Named Entity Barchart

import spacy
from collections import  Counter
import seaborn as sns

def plot_most_common_named_entity_barchart(text, entity="PERSON"):
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text,ent):
        doc=nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]

    entity_filtered=text.apply(lambda x: _get_ner(x,entity))
    entity_filtered=[i for x in entity_filtered for i in x]

    counter=Counter(entity_filtered)
    x,y=map(list,zip(*counter.most_common(10)))
    sns.barplot(x = y, y = x).set_title(entity)

plot_most_common_named_entity_barchart(news['headline_text'], entity="PERSON")


# * EXPLORATION THROUGH PARTS OF SPEECH TAGGING *
import nltk
nltk.download('averaged_perceptron_tagger')

sentence="The greatest comeback stories in 2019"
tokens=word_tokenize(sentence)
nltk.pos_tag(tokens)

import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
from collections import Counter

def plot_parts_of_speach_barchart(text):
    nltk.download('averaged_perceptron_tagger')

    def _get_pos(text):
        pos=nltk.pos_tag(word_tokenize(text))
        pos=list(map(list,zip(*pos)))[1]
        return pos

    tags=text.apply(lambda x : _get_pos(x))
    tags=[x for l in tags for x in l]
    counter=Counter(tags)
    x,y=list(map(list,zip(*counter.most_common(7))))

    sns.barplot(x=y,y=x)

plot_parts_of_speach_barchart(news['headline_text'])


# Code Snippet for Most Common Part of Speach Barchart

import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
from collections import Counter

def plot_most_common_part_of_speach_barchart(text, part_of_speach='NN'):
    nltk.download('averaged_perceptron_tagger')

    def _filter_pos(text):
        pos_type=[]
        pos=nltk.pos_tag(word_tokenize(text))
        for word,tag in pos:
            if tag==part_of_speach:
                pos_type.append(word)
        return pos_type


    words=text.apply(lambda x : _filter_pos(x))
    words=[x for l in words for x in l]
    counter=Counter(words)
    x,y=list(map(list,zip(*counter.most_common(7))))
    sns.barplot(x=y,y=x).set_title(part_of_speach)

plot_most_common_part_of_speach_barchart(news['headline_text'])

# * EXPLORING THROUGH TEXT COMPLEXITY *


# hey, setting the block comment to a variable named _ will prevent
# streamlit from showing everything in the block comment
# if u have a better way, lmk (I also don't want to manually select 
# the lines, then press "ctrl" + "/" )
_="""

# using pyLDAvis

# Code Snippet for Creating LDA visualization

import nltk
from nltk.corpus import stopwords

install("scipy==1.11.0")
# from numpy import triu
# from scipy.linalg import triu
# install("gensim")
install("git+https://github.com/piskvorky/gensim.git")
import gensim

from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim


def get_lda_objects(text):
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))


    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus

    corpus=_preprocess_text(text)

    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]

    lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = 4,
                                   id2word = dic,
                                   passes = 10,
                                   workers = 2)

    return lda_model, bow_corpus, dic

def plot_lda_vis(lda_model, bow_corpus, dic):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    return vis

nltk.download('wordnet')

# import threading

# download_thread = threading.Thread(target=get_lda_objects, name="LDA", args=news['headline_text'])
# download_thread.start()
lda_model, bow_corpus, dic = get_lda_objects(news['headline_text'])

"""