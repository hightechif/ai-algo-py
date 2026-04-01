import numpy as np
from scratch import NaiveBayesFromScratch
from framework import NaiveBayesFramework
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

def main() -> None:
    # 1. Data Preparation
    print("Loading 20 Newsgroups dataset...")
    categories = ['alt.atheism', 'comp.graphics']
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    X_train_text, y_train = train_data.data, train_data.target
    X_test_text, y_test = test_data.data, test_data.target

    print(f"Training samples: {len(X_train_text)}")
    print(f"Testing samples:  {len(X_test_text)}")

    # 2. Feature Extraction (Bag of Words)
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    print(f"Vocabulary size: {X_train.shape[1]}")

    # 3. Training & Comparison
    print("\nTraining NaiveBayesFromScratch...")
    nb_scratch = NaiveBayesFromScratch(alpha=1.0)
    nb_scratch.fit(X_train, y_train)

    print("Training NaiveBayesFramework...")
    nb_framework = NaiveBayesFramework(alpha=1.0)
    nb_framework.fit(X_train, y_train)

    # 4. Results
    scratch_acc = nb_scratch.score(X_test, y_test)
    framework_acc = nb_framework.score(X_test, y_test)

    print(f"\nResults:")
    print(f"Scratch Accuracy:   {scratch_acc:.4f}")
    print(f"Framework Accuracy: {framework_acc:.4f}")

    assert np.isclose(scratch_acc, framework_acc, atol=1e-2), "Accuracy mismatch between scratch and framework!"

if __name__ == "__main__":
    main()
