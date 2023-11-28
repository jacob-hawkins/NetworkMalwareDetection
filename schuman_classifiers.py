import os
import sys
import argparse
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, load_diabetes

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def setup_data(args):
    if (args.app_type == "load"):
        if (args.data_pkl is not None):
            with open(args.data_pkl, "rb") as f:
                X = pickle.load(f)
        elif (args.data_np is not None):
            X = np.load(args.data_np)
        else:
            raise Exception("When specifying app_type load, data and labels files must be specified with either data_pkl/labels_pkl or data_np/labels_np.")

        if (args.labels_pkl is not None):
            with open(args.labels_pkl, "rb") as f:
                y = pickle.load(f)
        elif (args.labels_np is not None):
            y = np.load(args.labels_np)
        else:
            raise Exception("When specifying app_type load, train and test files must be specified with either train_pkl/test_pkl or train_np/test_np.")

    elif (args.app_type == "std"):
        if (args.app_name == "iris"):
            X, y = load_iris(return_X_y=True)
        elif (args.app_name == "wine"):
            X, y = load_wine(return_X_y=True)
        elif (args.app_name == "digits"):
            X, y = load_digits(return_X_y=True)
        elif (args.app_name == "breast_cancer"):
            X, y = load_breast_cancer(return_X_y=True)
        elif (args.app_name == "diabetes"):
            X, y = load_diabetes(return_X_y=True)
        else:
            raise Exception("Std dataset specified is not implemented.")
    return X, y



names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
]

parser = argparse.ArgumentParser("All the classifiers!")
parser.add_argument("--app_type", default="std", choices=["std", "load"], help="application type")
parser.add_argument("--app_name", default="iris", type=str)
parser.add_argument("--split_test_size", default=0.33, type=float, help="Fraction of data to be used as testing data")
parser.add_argument("--split_seed", default=42, type=int, help="Seed used to determine the train/test split of the data")
parser.add_argument("--data_pkl", default=None, type=str, help="Pickle file with data")
parser.add_argument("--labels_pkl", default=None, type=str, help="Pickle file with labels")
parser.add_argument("--data_np", default=None, type=str, help="Numpy file with data")
parser.add_argument("--labels_np", default=None, type=str, help="Numpy file with labels")

args = parser.parse_args()

X, y = setup_data(args)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split_test_size, random_state=args.split_seed)

for i in range(len(classifiers)):
    cn = names[i]
    clf = classifiers[i]
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_predict)

    y_predict = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_predict)

    print("Name:", cn)
    print("Training:", train_acc)
    print("Testing:", test_acc)
    print("\n")   
 
