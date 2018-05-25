import tensorflow as tf
import pandas as pd
import numpy as np
from tfdnn.classifiers import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold

train = pd.read_csv('../data/iris/iris.data', header=None)
train.columns = ['sl', 'sw', 'pl', 'pw', 'label']
train = train.loc[train['label'] != 'Iris-setosa']
train.loc[train['label'] == 'Iris-versicolor', 'label'] = 0.
train.loc[train['label'] == 'Iris-virginica', 'label'] = 1.


x = train.values[:, :-1].astype(np.float32)
y = train.values[:, -1].reshape(-1, 1).astype(np.float32)

cls = LogisticRegression(optimizer=tf.train.GradientDescentOptimizer)

search = GridSearchCV(cls, {'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                            'l2_w': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                            'epochs': [100, 1000]},
                      cv=KFold(n_splits=10, shuffle=True))

search.fit(x, y)

print(search.best_score_)
print(search.best_params_)
print(search.best_estimator_)
