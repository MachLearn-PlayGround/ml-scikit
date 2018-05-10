"""
This file is a custom implementation of the Introduction to Machine Learning
using Scikit-Learn
Scikit data:
classification:
    iris
    digits
regression:
    boston house dataset
"""
from sklearn import datasets, svm


def load_data():
    """Use iris and digits standard datasets provided by scikit"""
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    return iris, digits


def predict_last(loaded_digits, last_predictions):
    """
    Learn from the training data except the last the use the last to test
    the prediction
    Using the digits datasets, predict given an image,
    which digit it represents
    Image is a 8 * 8 matrix

    sample results 0 - 9 == digits.target

    Use an estimator [uses fix(x, y) and predict(T)] to classify unseen samples
    
    :return: predicted value
    """
    # SVC implements Support Vector Classification
    # gamma and C are parameters of the model and in this case manually set
    classifier = svm.SVC(gamma=0.001, C=100.)  # The estimator

    # Next, this estimator should be fitted/learn to/from the model
    # Let's use the whole dataset as a training data except the last which
    # we'll predict
    classifier.fit(loaded_digits.data[:-1], loaded_digits.target[:-1])
    # predict target of last data
    predicted_values = classifier.predict(loaded_digits.data[-last_predictions:])
    actual_values = loaded_digits.target[-last_predictions:]
    return predicted_values, actual_values


if __name__ == "__main__":
    iris, digits = load_data()
    predicted, actual = predict_last(digits, 10)
    print(predicted, actual)
