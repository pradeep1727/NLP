from flask import Flask,render_template,url_for,request, send_file
from flask_bootstrap import Bootstrap 
from textblob import TextBlob,Word 
import csv
from csv import writer
import random 
import time
import pandas as pd 
from IPython import get_ipython
import joblib
import re
import nltk
import string
from textblob import TextBlob,Word
import numpy as np
from FinalCode import prediction
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.nlp.tokenizers import Tokenizer as sumytoken
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from flaskext.markdown import Markdown
from textblob import TextBlob,Word

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


app = Flask(__name__)
Markdown(app)
#Bootstrap(app)

LANGUAGE = "english"
#SENTENCES_COUNT = 5
stemmer = Stemmer(LANGUAGE)

def textsummary(text, stemmer, LANGUAGE, SENTENCES_COUNT):
    parser = PlaintextParser.from_string((text), sumytoken(LANGUAGE))
    summarizer_lsa = Summarizer(stemmer)
    summarizer_lsa.stop_words = get_stop_words(LANGUAGE)
    sentences = []
    for sentence in summarizer_lsa(parser.document, SENTENCES_COUNT):
        a = sentence
        sentences.append(str(a))
    return " ".join(sentences)

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def savedata(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'w', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


@app.route('/update')
def update():
	return render_template('update.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
	return render_template('upload.html')


@app.route('/analyse',methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        SENTENCES_COUNT = request.form['sentnum']
        blob = TextBlob(rawtext)
        blob_polarity = blob.sentiment.polarity
        blob_subjectivity = blob.sentiment.subjectivity
        summary = textsummary(rawtext, stemmer, LANGUAGE, SENTENCES_COUNT)
        tag = prediction(rawtext)
        listToStr = ' '.join([str(elem) for elem in tag])
        listToStr = listToStr.replace("'", "")
        listToStr = re.sub(r'[.|)|(|\|/]',r' ',listToStr)
        row_contents1 = [rawtext,listToStr]
        savedata('predictedtext.csv', row_contents1)

    return render_template('result.html',received_text = rawtext,summary=summary,tag=listToStr, polarity= blob_polarity, subjectivity = blob_subjectivity)


@app.route('/data1', methods=['GET', 'POST'])
def data1():
    if request.method == 'POST':
        f = request.form['csvfile']
        df = []
        df1 = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                row = ' '.join([str(elem) for elem in row]) 
                x =prediction(row)
                df.append(x)
                df1.append(row)
        df = pd.DataFrame(df)
        df1 = pd.DataFrame(df1)
        df2 = pd.concat([df1,df], axis=1)
        with open('my_csv.csv', 'w') as f:
            df2.to_csv('download.csv', header=False, index=False)
    return render_template('data1.html', data1=df2.to_html(header=False, index = False))

@app.route('/download', methods=['GET', 'POST'])
def download():
    return send_file("download.csv", as_attachment = True)



@app.route('/addtag',methods=['GET', 'POST'])
def addtag():
    if request.method == 'POST':
        feedback = request.form['complaint']
        topic = request.form['tag']
        row_contents = [feedback,topic]
        append_list_as_row('test.csv', row_contents)
    return render_template('preview.html', feedback = feedback,topic=topic)

@app.route('/adddata')
def adddata():
    a= pd.read_csv("predictedtext.csv")
    append_list_as_row('test.csv', a)
    return render_template('add.html')



if __name__ == '__main__':
    app.run(debug=True)