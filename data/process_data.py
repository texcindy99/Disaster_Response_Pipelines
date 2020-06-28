# imports
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    function: 
    Load message and categories dataset from csv file

    input:
    messages_filepath: file path of messages data file
    categories_filepath: file path of categoreis data file

    output:
    df: 2D dataframe after merging messages and categoreis data
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df
    
def clean_data(df):
    """
    function:
    Clean input dataframe by numeric coding and removing duplicates 

    input:
    df: 2D dataframe

    output:
    df: 2D dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        ## categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
     
    # drop duplicates 
    df.drop_duplicates(subset='message',inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    function:
    Save dataframe to database

    input:
    df: 2D dataframe
    database_filename: name of data file to be save in the database

    output:
    The data file saved in the database
    """
    # Get the data table name
    path_str=database_filename.split('/')
    for name in path_str :
        if name.find('.db'):
            datatable_name=name.split('.')[0]
            
    # Establish the connection the the database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(datatable_name, engine, if_exists='replace', index=False)
    conn = engine.raw_connection()
    
    # commit any changes to the database and close the connection to the    database
    conn.commit()
    conn.close()

def main():
    """
    main function:
    Calling functions to load data files, clean and save data to database
    
    input: None
    
    output: data file in the database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()