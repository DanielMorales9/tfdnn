import tensorflow as tf
import pandas as pd
import numpy as np
from tfdnn.classifiers import DeepNeuralNetwork
from sklearn.preprocessing import StandardScaler, OneHotEncoder

train = pd.read_csv('../data/iris/iris.data', header=None)
train.columns = ['sl', 'sw', 'pl', 'pw', 'label']
# train = train.loc[train['label']!='Iris-virginica']
train.loc[train['label'] == 'Iris-setosa', 'label'] = 0.
train.loc[train['label'] == 'Iris-versicolor', 'label'] = 1.
train.loc[train['label'] == 'Iris-virginica', 'label'] = 2.


enc = OneHotEncoder(sparse=False, dtype=np.float64)
y = train.values[:, -1].reshape(-1, 1).astype(np.float32)
y = enc.fit(y).transform(y)

x = train.values[:, :-1].astype(np.float32)

cls = DeepNeuralNetwork(optimizer=tf.train.AdamOptimizer,
                        learning_rate=0.001,
                        batch_size=10,
                        act_fun='relu',
                        loss_function='softmax_cross_entropy',
                        hidden_units=[80, 20, 10],
                        epochs=1000, regularization=0.0)

x = StandardScaler().fit_transform(x)
cls.fit(x, y)
c = np.zeros((x.shape[0], 2))
y_hat = cls.predict(x)
print(y_hat)

c[:, 0] = np.argmax(y, axis=1)
c[:, 1] = np.argmax(y_hat, axis=1)
print(c)
print(np.sum(y_hat, axis=1))
print(cls.score(x, y))

