import json

import flask

from flask import render_template

from collections import defaultdict

from gensim import corpora, models, similarities

import flask

from flask import request

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from nltk.corpus import stopwords

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it

spark_df = pd.read_csv("sparks.csv", index_col=0)

book_cliff=pd.read_csv('cliff_df.csv',index_col=0)

book_df=pd.read_csv('book.csv',index_col=0)



book_cliff['descriptions']=book_cliff['descriptions'].astype(str)
book_df['syn']=book_df['syn'].astype(str)



good_titles=list(book_df['title'])
good_descriptions=list(book_df['syn'])


spark_titles=list(spark_df['spark_title'])
spark_descriptions=list(spark_df['spark_summary'])

cliff_titles=list(book_cliff['title'])
cliff_descriptions=list(book_cliff['descriptions'])

stop_books = stopwords.words('english')
stop_books += ['.', ',', '(', ')', "'", '"','novel','books',
               'published','characters','works','mr.','ms.',u"'s",u"time",u"only",u"n't",u"book",u'life',u'one',
          u'story',u'one',u'story',u'tells',u'things','..',"&","must",'new','like',"she's","he's","gets","-","--","/",":","shes","hes",
              "get","like","--the",'"the','says','two','him.']

texts=[[word for word in document.lower().split() 
        if word not in stop_books] for document in good_descriptions]

frequency=defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1

texts=[[token for token in text if frequency[token]>1] for text in texts]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

lsi = models.LsiModel(corpus, id2word=dictionary,num_topics=10) #num_topics=
tfidf=models.TfidfModel(corpus,id2word=dictionary)
lda = models.LdaModel(corpus,id2word=dictionary,passes=5,num_topics=4) #num_topics=



def recommender(title):
    index_n=good_titles.index(title)
    document=good_descriptions[index_n]
    search_vec = dictionary.doc2bow(document.lower().split())


    vec_lsi = lsi[search_vec]
    index_lsi = similarities.MatrixSimilarity(lsi[corpus])
    sims_lsi = index_lsi[vec_lsi]

    vec_tfidf=tfidf[search_vec]
    index_tfidf = similarities.MatrixSimilarity(tfidf[corpus])
    sims_tfidf=index_tfidf[vec_tfidf]

    vec_lda=lda[search_vec]
    index_lda=similarities.MatrixSimilarity(lda[corpus])
    sims_lda=index_lda[vec_lda]

    combined=(sims_lsi+sims_lda*.5+sims_tfidf)/3

    df_rec=pd.DataFrame()
	
    df_rec['lda']=sims_lda
    df_rec['lsi']=sims_lsi
    df_rec['tfidf']=sims_tfidf

    df_norm = (df_rec - df_rec.mean()) / (df_rec.max() - df_rec.min())
    df_norm['total']=(df_norm['lda']+df_norm['lsi']+df_norm['tfidf'])/3
    df_norm['title']=good_titles

    df_norm=df_norm.sort_values(by="total",ascending=False)
    return df_norm

def bsearch(paragraph):
    search_vec = dictionary.doc2bow(paragraph.lower().split())

    vec_lsi = lsi[search_vec]
    index_lsi = similarities.MatrixSimilarity(lsi[corpus])
    sims_lsi = index_lsi[vec_lsi]

    vec_tfidf=tfidf[search_vec]
    index_tfidf = similarities.MatrixSimilarity(tfidf[corpus])
    sims_tfidf=index_tfidf[vec_tfidf]

    vec_lda=lda[search_vec]
    index_lda=similarities.MatrixSimilarity(lda[corpus])
    sims_lda=index_lda[vec_lda]

    combined=(sims_lsi+sims_lda+sims_tfidf)/3

    df_rec=pd.DataFrame()
    
    df_rec['lda']=sims_lda
    df_rec['lsi']=sims_lsi
    df_rec['tfidf']=sims_tfidf

    df_norm = (df_rec - df_rec.mean()) / (df_rec.max() - df_rec.min())
    df_norm['total']=(df_norm['lda']*.5+df_norm['lsi']+df_norm['tfidf'])/3
    df_norm['title']=good_titles

    df_norm=df_norm.sort_values(by="total",ascending=False)
    return df_norm


def sum_spark(doc):

    parser = PlaintextParser.from_string(doc,Tokenizer('english'))

    summarizer = Summarizer(Stemmer('english'))
    summarizer.stop_words = stop_books
    
    texts=[]

    for sentence in summarizer(parser.document, 2):
        texts.append(str(sentence))

    return texts


def line_break(summary):
    lines=[]
    for sentence in summary.split('.'):
        lines.append(str(sentence))
    return lines


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage

@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, 
    """
    return render_template('fletchertest.html')



# Get an example and return it's score from the predictor model
@app.route("/nlp", methods=["POST"])
def nlp():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    # Get decision score for our example that came with the request
    data = flask.request.json
    #x = data["example"]
    title=str(data)
    index_n=spark_titles.index(title)
    document=spark_descriptions[index_n]
    #x = str(data)
    df=recommender(title)
    titles=list(df['title'])
    total_scores=list(df['total'])
    lda_scores=list(df['lda'])
    lsi_scores=list(df['lsi'])
    tfidf_scores=list(df['tfidf'])
    short=sum_spark(document)
    t0=titles[0]
    t1=titles[1]
    t2=titles[2]
    t3=titles[3]
    t4=titles[4]
    t5=titles[5]
    s0=int(total_scores[0]*100)
    s1=int(total_scores[1]*100)
    s2=int(total_scores[2]*100)
    s3=int(total_scores[3]*100)
    s4=int(total_scores[4]*100)
    s5=int(total_scores[5]*100)
    d0=int(lda_scores[0]*100)    
    d1=int(lda_scores[1]*100)
    d2=int(lda_scores[2]*100)
    d3=int(lda_scores[3]*100)
    d4=int(lda_scores[4]*100)
    d5=int(lda_scores[5]*100)
    l0=int(lsi_scores[0]*100)
    l1=int(lsi_scores[1]*100)
    l2=int(lsi_scores[2]*100)
    l3=int(lsi_scores[3]*100)
    l4=int(lsi_scores[4]*100)
    l5=int(lsi_scores[5]*100)
    f0=int(tfidf_scores[0]*100)
    f1=int(tfidf_scores[1]*100)
    f2=int(tfidf_scores[2]*100)
    f3=int(tfidf_scores[3]*100)
    f4=int(tfidf_scores[4]*100)
    f5=int(tfidf_scores[5]*100)

    #x = data["example"]
    #par=line_break(document)
    #summ=summurize(title)
    #score = PREDICTOR.predict_proba(x)
    #Put the result in a nice dict so we can send it as json
    #return JSON.stringify(summ)
    results = {"summary": document,'short':short,
    't0':t0,'t1':t1,'t2':t2,'t3':t3,'t4':t4,'t5':t5,
    's0':s0,'s1':s1,'s2':s2,'s3':s3,'s4':s4,'s5':s5,
    'd0':d0,'d1':d1,'d2':d2,'d3':d3,'d4':d4,'d5':d5,
    'l0':l0,'l1':l1,'l2':l2,'l3':l3,'l4':l4,'l5':l5,
    'f0':f0,'f1':f1,'f2':f2,'f3':f3,'f4':f4,'f5':f5}
    return flask.jsonify(results)

@app.route("/searcher", methods=["POST"])
def searcher():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    # Get decision score for our example that came with the request
    data = flask.request.json
    #x = data["example"]
    title=str(data)
    #x = str(data)
    df=bsearch(title)
    titles=list(df['title'])
    total_scores=list(df['total'])
    lda_scores=list(df['lda'])
    lsi_scores=list(df['lsi'])
    tfidf_scores=list(df['tfidf'])
    short=sum_spark(document)
    t0=titles[0]
    t1=titles[1]
    t2=titles[2]
    t3=titles[3]
    t4=titles[4]
    t5=titles[5]
    s0=int(total_scores[0]*100)
    s1=int(total_scores[1]*100)
    s2=int(total_scores[2]*100)
    s3=int(total_scores[3]*100)
    s4=int(total_scores[4]*100)
    s5=int(total_scores[5]*100)
    d0=int(lda_scores[0]*100)    
    d1=int(lda_scores[1]*100)
    d2=int(lda_scores[2]*100)
    d3=int(lda_scores[3]*100)
    d4=int(lda_scores[4]*100)
    d5=int(lda_scores[5]*100)
    l0=int(lsi_scores[0]*100)
    l1=int(lsi_scores[1]*100)
    l2=int(lsi_scores[2]*100)
    l3=int(lsi_scores[3]*100)
    l4=int(lsi_scores[4]*100)
    l5=int(lsi_scores[5]*100)
    f0=int(tfidf_scores[0]*100)
    f1=int(tfidf_scores[1]*100)
    f2=int(tfidf_scores[2]*100)
    f3=int(tfidf_scores[3]*100)
    f4=int(tfidf_scores[4]*100)
    f5=int(tfidf_scores[5]*100)

    #x = data["example"]
    #par=line_break(document)
    #summ=summurize(title)
    #score = PREDICTOR.predict_proba(x)
    #Put the result in a nice dict so we can send it as json
    #return JSON.stringify(summ)
    results = {"summary": document,'short':short,
    't0':t0,'t1':t1,'t2':t2,'t3':t3,'t4':t4,'t5':t5,
    's0':s0,'s1':s1,'s2':s2,'s3':s3,'s4':s4,'s5':s5,
    'd0':d0,'d1':d1,'d2':d2,'d3':d3,'d4':d4,'d5':d5,
    'l0':l0,'l1':l1,'l2':l2,'l3':l3,'l4':l4,'l5':l5,
    'f0':f0,'f1':f1,'f2':f2,'f3':f3,'f4':f4,'f5':f5}
    return flask.jsonify(results)

@app.route("/sum_text", methods=["POST"])
def sum_text():
    data = flask.request.json
    #x = data["example"]
    #document=str(data)
    short=sum_spark(data)
    results = {"summary": short}
    return flask.jsonify(results)


@app.route("/good", methods=["POST"])
def good():
    data = flask.request.json
    title=str(data)
    index_n=good_titles.index(title)
    document=good_descriptions[index_n]
    #document=str(data)
    results = {"summary": document}
    return flask.jsonify(results)


@app.route("/cliff", methods=["POST"])
def cliff():
    data = flask.request.json
    title=str(data)
    index_n=cliff_titles.index(title)
    document=cliff_descriptions[index_n]
    #document=str(data)
    results = {"summary": document}
    return flask.jsonify(results)

@app.route("/spark", methods=["POST"])
def spark():
    data = flask.request.json
    title=str(data)
    index_n=spark_titles.index(title)
    document=spark_descriptions[index_n]
    #document=str(data)
    results = {"summary": document}
    return flask.jsonify(results)



#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=5000,debug=True)