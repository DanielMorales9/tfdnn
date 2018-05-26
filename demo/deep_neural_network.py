import tensorflow as tf
import pandas as pd
import numpy as np
from tfdnn.classifiers import DeepNeuralNetwork
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../data/iris/iris.data', header=None)
train.columns = ['sl', 'sw', 'pl', 'pw', 'label']
train = train.loc[train['label'] != 'Iris-setosa']
train.loc[train['label'] == 'Iris-versicolor', 'label'] = 0.
train.loc[train['label'] == 'Iris-virginica', 'label'] = 1.


x = train.values[:, :-1].astype(np.float32)
y = train.values[:, -1].reshape(-1, 1).astype(np.float32)

cls = DeepNeuralNetwork(optimizer=tf.train.GradientDescentOptimizer,
                        learning_rate=1, hidden_units=[80, 10], batch_norm=True,
                        epochs=1000, regularization=0.)

x = StandardScaler().fit_transform(x)
cls.fit(x, y)
y_hat = (cls.predict(x) > 0.5).astype(np.float)
c = np.zeros((y_hat.shape[0], 2))

c[:, 0] = y.reshape(-1)
c[:, 1] = y_hat.reshape(-1)

print(c)

print(cls.score(x, y))

