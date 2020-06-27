# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    """
    function: 
    Load data from database
    
    input:
    database_filepath: file name and path of data table
    
    output:
    X: 1D dataframe containing messages
    Y: 2D dataframe containing numeric category labels
    category_names: List of category names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # check table names in the database
    print(engine.table_names())
    
    # find the data table name
    path_str=database_filepath.split('/')
    for name in path_str :
        if name.find('.db'):
            datatable_name=name.split('.')[0]
    print(datatable_name)
    
    # read table from database
    df = pd.read_sql_table(datatable_name, engine)
    
    # close the connection to the database
    conn = engine.raw_connection()
    conn.close()
    
    # Extract X and Y
    X = df['message']
    # Drop non-category features
    Y = df.drop(['id', 'message','original', 'genre'], axis = 1)
    # Replace "2" in Y to "1" to simplify classification
    # Y[Y>=2]=1
    Y=Y.replace(to_replace ='2', value ='0') 
    Y=Y.astype(int)
    
    # Get category names
    category_names = Y.columns
    
    return X, Y, category_names

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

def display_results(cv, y_test, y_pred, category_names):
    """
    function:
    print test score for classification model
    
    input:
    cv: classification model with best parameter from GridSearchCV
    y_test: actual label for test data
    y_pred: predicted label for test data
    category_names: category names for y label
    
    output:
    print out test score and best parameters for model
    """
    # Accumulate the test score for each category
    score=[]
    for i in range(y_pred.shape[1]):
        score.append(precision_recall_fscore_support(y_test.iloc[:,i], y_pred[:,i], average='macro')[0:3])
        # Print out score for all categories
        print('Category: {} \n'.format(category_names[i]))
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))
            
    # Calculate weighted score for each category and sum them up as and average score
    # Calculate weights based on pertentage of "1" label in each category
    weights=(y_test[y_test>0].count())/(y_test[y_test>0].count().sum())
    score_weight=[]
    for i in range(len(score)):
        score_weight.append(pd.DataFrame(score).iloc[i,:].apply(lambda x: x*weights[i]).values)
    score_Avg=sum(score_weight)
    
    # Print out model if from GridSearch
    print("\nBest Parameters:", cv.best_params_)

    # print out average score    
    print('Model Average Score [precision, recall, f1-score]={}'.format(score_Avg)) 

def build_model():
    """
    function:
    build pipeline with GridSearchCV
    
    input: None
    
    output: 
    cv: pipeline model with GridSearchCV
    """
    # Build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Set parameter grid for optimization
    parameters = {
    'features__text_pipeline__vect__max_df': (0.5, 1.0),
    'clf__estimator__n_estimators': [20, 50]
    }

    # create grid search object
    cv = GridSearchCV(pipeline,param_grid=parameters,scoring='f1_macro',cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    function:
    Evaluate pipeline with GridSearchCV on test dataset
    
    input: 
    model: pipeline model with best parameters from GridSearchCV
    X_test: 2D dataframe of test dataset features
    Y_test: 2D dataframe of test dataset labels
    category_names: list of category names of Y_test labels
    
    output: 
    print out test results
    """
    y_pred = model.predict(X_test)
    
    display_results(model,Y_test, y_pred,category_names)


def save_model(model, model_filepath):
    """
    function:
    Save best pipeline model to pickle file
    
    input: 
    model: pipeline model with best parameters from GridSearchCV
    model_filepath: path and name of pickle file to be saved
    
    output: 
    saved pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """
    main function:
    Load data, build model, train model, test model with best parameters and save model to pickle file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
          
if __name__ == '__main__':
    main()