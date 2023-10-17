import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Please run `python -m spacy download en`

"""
This script takes a raw text as input and outputs a list of cleaned, tokenized words. It performs tokenization, lowercasing, punctuation removal, stopword removal, and either stemming (for NLTK) or lemmatization (for SpaCy).

Remember, this is a very basic form of preprocessing. Depending on your specific task and dataset, you might need to add more steps (such as handling special characters, correcting spelling, etc.) or modify the existing ones.
"""

# Downloading NLTK stopwords and punkt tokenizer
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_with_nltk(raw_text):
    # Tokenization
    words = word_tokenize(raw_text)
    print("Tokenized words:", words)

    # Lowercasing
    words = [word.lower() for word in words]
    print("Lowercased words:", words)

    # Removing punctuation
    words = [word for word in words if word.isalnum()]
    print("Alphanumeric words:", words)

    # Stopword removal
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    print("Stopword-free words:", words)

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    print("Stemmed words:", words)

    return words


def preprocess_with_spacy(raw_text):
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(raw_text)
    # Non-stop
    words = [token for token in doc if not token.is_stop]
    print("Stopword-free words:", words)

    # Alpha
    words = [token for token in words if token.is_alpha]
    print("Alphanumeric words:", words)

    # Lemmatization
    words = [token.lemma_ for token in words]
    print("Lemmatized words:", words)

    # Tokenization, lowercasing, punctuation removal, stopword removal, and lemmatization
    doc = nlp(raw_text)
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    print("Fully preprocessed words:", words)

    return words

def main():
    print("________________ Now it begins _______________")
    # Test the functions with a sample text
    raw_text = "Hello, world! This is a sample text. It's for testing text preprocessing."
    print(raw_text)

    print("NLTK:")
    print(preprocess_with_nltk(raw_text))

    print("SpaCy:")
    print(preprocess_with_spacy(raw_text))

if __name__ == '__main__':
    main()
