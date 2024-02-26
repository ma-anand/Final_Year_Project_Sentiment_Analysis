import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Downloading StopWords
import nltk
nltk.download('stopwords')
#print(stopwords.words('english'))


#Data Processing

#loading the data from csv file to pandas dataframe
user_data=pd.read_csv('fileName.csv', encoding= 'ISO-8859-1')

user_data.shape