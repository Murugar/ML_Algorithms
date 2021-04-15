import os
from pandas import read_csv, concat
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# Load data
data_path = os.path.join(os.getcwd(), "titanic.csv")
dataset = read_csv(data_path, skipinitialspace=True)

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
title_mapping = { 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5 }
gender_mapping = { 'female': 1, 'male': 0 }
port_mapping = { 'S': 0, 'C': 1, 'Q': 2 }

# Map title
dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)

# Map gender
dataset['Sex'] = dataset['Sex'].map(gender_mapping).astype(int)

# Map port
freq_port = dataset.Embarked.dropna().mode()[0]
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
dataset['Embarked'] = dataset['Embarked'].map(port_mapping).astype(int)

# Fix missing age values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].dropna().median())

X = dataset.drop(['Survived'], axis = 1).values
Y = dataset[['Survived']].values

X = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = None)

# Prepare cross-validation (cv)
cv = KFold(n_splits = 5, random_state = None)

# Performance
p_score = lambda model, score: print('Performance of the %s model is %0.2f%%' % (model, score * 100))

# Classifiers
names = [
    "Logistic Regression", "Logistic Regression with Polynomial Hypotheses",
    "Linear Support Vector Machines", "Radial Basis Function Support Vector Machines", "Neural Networks",
]

classifiers = [
    LogisticRegression(),
    make_pipeline(PolynomialFeatures(3), LogisticRegression()),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    MLPClassifier(alpha=1),
]

# iterate over classifiers
models = []
trained_classifiers = []
for name, clf in zip(names, classifiers):
    scores = []
    for train_indices, test_indices in cv.split(X):
        clf.fit(X[train_indices], Y[train_indices].ravel())
        scores.append( clf.score(X_test, Y_test.ravel()) )
    
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    trained_classifiers.append(clf)
    models.append((name, min_score, max_score, avg_score))
    
fin_models = pd.DataFrame(models, columns = ['Algorithm', 'Minimum', 'Maximum', 'Mean'])

print(fin_models.sort_values(['Mean']).head())




