from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split

"""
Use the popular 20 Newsgroups dataset that comes with Scikit-learn. It's a collection of approximately 20,000 newsgroup documents,
partitioned (nearly) evenly across 20 different newsgroups. It's a good dataset for text classification.

Here is a basic outline of a Naive Bayes classifier for text classification using Bag-of-Words (BoW) features:
"""

# Load the dataset
newsgroups_data = fetch_20newsgroups(subset='all')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups_data.data, newsgroups_data.target, test_size=0.2, random_state=42)

# Transform the text data into vectors using CountVectorizer (BoW model)
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectors, y_train)

# Predict on the test data
y_pred = clf.predict(X_test_vectors)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
