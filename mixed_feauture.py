import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# Download stopwords
nltk.download('stopwords')

# Download Word2Vec pre-trained model
nltk.download('punkt')

df = pd.read_csv("D:\\B. Tech\\2nd Year\\4th Semester\\PBL\\MODEL_FINAL\\fake reviews dataset.csv")

# Your preprocessing steps here...
df = pd.read_csv('Preprocessed Fake Reviews Detection Dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)
df['length'] = df['text_'].apply(len)
df.info()

plt.hist(df['length'], bins=50)
plt.show()

df.groupby('label').describe()
df.hist(column='length', by='label', bins=50, color='blue', figsize=(12, 5))
plt.show()

df[df['label'] == 'OR'][['text_', 'length']].sort_values(by='length', ascending=False).head().iloc[0].text_
df.length.describe()


def preprocess_text(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# Assuming df['text_'] contains NaN values
df['text_'].fillna('', inplace=True)

# Tokenize and preprocess the entire dataset
tokenized_text = df['text_'].apply(preprocess_text)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)


def get_average_vector(tokens, model, vector_size):
    valid_tokens = [token for token in tokens if token in model.wv]
    if valid_tokens:
        return np.mean(model.wv[valid_tokens], axis=0)
    else:
        return np.zeros(vector_size)



# Apply the function to get the average vector for each document
word2vec_features = tokenized_text.apply(lambda x: get_average_vector(x, word2vec_model, 100))

# Concatenate the word2vec features with other features if any
final_features_word2vec = pd.concat([pd.DataFrame(word2vec_features.tolist(), index=df.index),
                                    df[['length']].astype(str)], axis=1)

# Explicitly set column names as strings
final_features_word2vec.columns = final_features_word2vec.columns.astype(str)
final_features_word2vec['length'] = final_features_word2vec['length'].astype(str)


# Custom transformer for selecting specific columns from DataFrame
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Custom transformer for Word2Vec features
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: get_average_vector(x, self.word2vec_model, 100))




# BoW vectorizer
bow_vectorizer = CountVectorizer(analyzer=preprocess_text)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer=preprocess_text, dtype=np.float64)

# Custom transformer instances
word2vec_transformer = Word2VecTransformer(word2vec_model)
df_selector = DataFrameSelector(['text_'])

# Feature union
features_union = FeatureUnion([
    ('bow', bow_vectorizer),
    ('tfidf', tfidf_vectorizer),
    ('word2vec', word2vec_transformer)
])

# Concatenate the word2vec features with other features if any
final_features_combined = features_union.fit_transform(df_selector.transform(df))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(final_features_combined, df['label'], test_size=0.35, random_state=42)

# List of classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier())
]

# Iterate through classifiers
for clf_name, clf in classifiers:
    # Create a pipeline with feature scaling if needed
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features if needed
        ('classifier', clf)
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predictions on the test set
    predictions = pipeline.predict(X_test)

    # Evaluate the classifier
    print(f'{clf_name}:')
    print('Classification Report:', classification_report(y_test, predictions))
    print('Confusion Matrix:', confusion_matrix(y_test, predictions))
    print('Accuracy Score:', accuracy_score(y_test, predictions))
    print('\n' + '-'*50 + '\n')
