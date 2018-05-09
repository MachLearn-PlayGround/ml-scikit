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
from sklearn import datasets


def load_data():
    """Use iris and digits standard datasets provided by scikit"""
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    return iris, digits


if __name__ == "__main__":
    a, b = load_data()
    print(a)
