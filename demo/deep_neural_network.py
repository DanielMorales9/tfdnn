import tensorflow as tf
import pandas as pd
import numpy as np
from tfdnn.classifiers import DeepNeuralNetwork
from sklearn.preprocessing import StandardScaler, OneHotEncoder

train = pd.read_csv('../data/iris/iris.data', header=None)
train.columns = ['sl', 'sw', 'pl', 'pw', 'label']
train = train.loc[train['label']!='Iris-virginica']
train.loc[train['label'] == 'Iris-setosa', 'label'] = 0.
train.loc[train['label'] == 'Iris-versicolor', 'label'] = 1.
# train.loc[train['label'] == 'Iris-virginica', 'label'] = 2.


# enc = OneHotEncoder(sparse=False, dtype=np.float64)
y = train.values[:, -1].reshape(-1, 1).astype(np.float32)
# y = enc.fit(y).transform(y)

x = train.values[:, :-1].astype(np.float32)

cls = DeepNeuralNetwork(optimizer=tf.train.AdamOptimizer,
                        learning_rate=0.0001,
                        act_fun='relu',
                        loss_function='cross_entropy',
                        hidden_units=[80, 20, 10],
                        epochs=1000, regularization=0.0)

x = StandardScaler().fit_transform(x)
cls.fit(x, y)
c = np.zeros((x.shape[0], 2))
y_hat = (cls.predict(x) > 0.5).astype(np.float32)

c[:, 0] = y.reshape(-1)
c[:, 1] = y_hat.reshape(-1)
print(c)
print(cls.score(x, y))

