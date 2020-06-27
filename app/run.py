import json
import plotly
import pandas as pd
import numpy as np

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    function: 
    Tokenize text

    input:
    text: text string

    output:
    clean_tokens: list of clean text tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Only consider word which is alphanumeric and not in stopwords
    tokens = [ w for w in word_tokenize(text) if (w.isalnum()) &
              (w not in nltk.corpus.stopwords.words("english"))
              ]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    class: part-of-speach tagging
    """

    def starting_verb(self, text):
        """
        function:
        Tokenize text and assign numeric tags 0/1 to each token
        
        input: 
        text: text string
        
        output:
        0/1: integer tag
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))   # part-of-speech tagging
            if pos_tags != []:  # skip one-letter text with no tag
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, x, y=None):
        """
        function:
        fit StartingVerbExtractor model
        
        """
        return self

    def transform(self, X):
        """
        function:
        transform text input into numeric tags in dataframe
        
        input:
        X: dataframe containing text
        
        output:
        X_tagged: dataframe containing numeric tags
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    ## TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ## data for 2nd plot
    Y = df.drop(['id', 'message','original', 'genre'], axis = 1)
    Y=Y.replace(to_replace ='2', value ='0') 
    Y=Y.astype(int)
    # Get "1" count for each category
    cat_count = Y.sum().sort_values(ascending=False)
    cat_name = list(cat_count.index) 
    
    ## data for 3rd plot
    hist, bin_edges = np.histogram(Y.sum(axis=1),bins=np.arange(start=0.5, stop=21.5, step=1))
    bin_grid=np.arange(start=1, stop=21, step=1)
    
    ## data for 4th plot
    # Get list of word tokens and counts
    X_part=[]
    X_part=[text for text in df.iloc[0:100,:]['message']]
    #X_part=[text for text in df[df['aid_related'].str.contains('1')]['message']] # take too long to render
    # print(X_part[0:2])
    vect = CountVectorizer(tokenizer=tokenize)
    X_vectorized = vect.fit_transform(X_part)
    word_list = vect.get_feature_names();    
    count_list = X_vectorized.toarray().sum(axis=0) 
    
    top10_index=np.argsort(count_list)[::-1][:10] 
    top10_word=[word_list[i] for i in top10_index]
    top10_count=[count_list[i] for i in top10_index]
   
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # fig 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # fig 2
        {
            'data': [
                Bar(
                    x=cat_name[0:10],
                    y=cat_count[0:10]
                )
            ],

            'layout': {
                'title': 'Top 10 Category Counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
         # fig 3
        {
            'data': [
                Bar(
                    x=bin_grid,
                    y=hist
                )
            ],

            'layout': {
                'title': 'Distribution of Number of lables for Each Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of lables for Each Message"
                }
            }
        },
        
        # fig 4
        {
            'data': [
                Bar(
                    x=top10_word,
                    y=top10_count
                )
            ],

            'layout': {
                'title': 'Counts of Top 10 words in first {} messages'.format(len(X_part)),
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()