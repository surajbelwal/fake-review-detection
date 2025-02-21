import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import Word2Vec

# Download stopwords
nltk.download('stopwords')

# Download Word2Vec pre-trained model
nltk.download('punkt')

df = pd.read_csv("D:\\B. Tech\\2nd Year\\4th Semester\\PBL\\MODEL_FINAL\\fake reviews dataset.csv")

# Your preprocessing steps here...
df = pd.read_csv('Preprocessed Fake Reviews Detection Dataset.csv')
df.head()
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
df.dropna(inplace=True)
df['length'] = df['text_'].apply(len)
df.info()
plt.hist(df['length'],bins=50)
plt.show()
df.groupby('label').describe()
df.hist(column='length',by='label',bins=50,color='blue',figsize=(12,5))
plt.show()
df[df['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_
df.length.describe()
def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Assuming df['text_'] contains NaN values
df['text_'].fillna('', inplace=True)

# Function to tokenize and preprocess text
def preprocess_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in word_tokenize(nopunc.lower()) if word not in stopwords.words('english')]

# Tokenize and preprocess the entire dataset
tokenized_text = df['text_'].apply(preprocess_text)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Function to calculate the average vector for a document
def get_average_vector(tokens, model, vector_size):
    if len(tokens) > 0:
        return np.mean(model.wv[tokens], axis=0)
    else:
        return np.zeros(vector_size)

# Apply the function to get the average vector for each document
word2vec_features = tokenized_text.apply(lambda x: get_average_vector(x, word2vec_model, 100))

# Concatenate the word2vec features with other features if any
final_features = pd.concat([pd.DataFrame(word2vec_features.tolist(), index=df.index), df[['length']].astype(str)], axis=1)

# Explicitly set column names as strings
final_features.columns = final_features.columns.astype(str)
final_features['length'] = final_features['length'].astype(str)





# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(final_features, df['label'], test_size=0.35, random_state=42)

# Your classification pipeline here...
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler






# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(final_features, df['label'], test_size=0.35, random_state=42)


# List of classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
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