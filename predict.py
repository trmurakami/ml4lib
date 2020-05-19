import sys
import pandas as pd
import re
import nltk
from sklearn.datasets import load_files
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np

titulosTeses = pd.read_csv('http://tecbib.com/FD-ECA.tsv',sep="\t")

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=stopwords.words('portuguese'))

X_train, X_test, y_train, y_test = train_test_split(titulosTeses['titulo'], titulosTeses['unidade'], test_size=0.30, random_state=42)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf_vectorize = tfidf_transformer.fit_transform(X_train_counts)

joblib_file = "clf.pkl"
try:
    f = open(joblib_file)
    f.close()
    clf = joblib.load(joblib_file)
except FileNotFoundError:
    print('File does not exist')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(70, ), random_state=1, max_iter=200, verbose=True)
    clf.fit(X_train_tfidf_vectorize, y_train)
    joblib.dump(clf, joblib_file)

print(clf.predict(count_vect.transform([sys.argv[1]])))