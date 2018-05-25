# tfdnn

This is a library for Deep Neural Network written in [TensorFlow™](https://www.tensorflow.org/).  
The library provides Logistic Regression, Shallow Neural Networks and Deep Neural Networks.


### What are Neural Networks?
Neural Networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.

![nn](./images/nn.png "Neural Networks")

```TODO```

## Usage

```tfdnn``` classifiers implement 
```scikit-learn```'s interface, thus ```tfdnn``` 
is compatible with any ```scikit-learn``` tool.  
Below we show a demo on how to use the ```tfdnn``` library. 

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from tfdnn.classifiers import DeepNeuralNetwork

train = pd.read_csv('../data/iris/iris.data', header=None)
train.columns = ['sl', 'sw', 'pl', 'pw', 'label']
train = train.loc[train['label'] != 'Iris-setosa']
train.loc[train['label'] == 'Iris-versicolor', 'label'] = 0.
train.loc[train['label'] == 'Iris-virginica', 'label'] = 1.


x = train.values[:, :-1].astype(np.float32)
y = train.values[:, -1].reshape(-1, 1).astype(np.float32)

cls = DeepNeuralNetwork(optimizer=tf.train.GradientDescentOptimizer,
                        learning_rate=0.01, hidden_units=[10, 10],
                        epochs=1000)

cls.fit(x, y)
y_hat = (cls.predict(x) > 0.5).astype(np.float)
c = np.zeros((y_hat.shape[0], 2))

print(cls.predict(x))
c[:, 0] = y.reshape(-1)
c[:, 1] = y_hat.reshape(-1)

print(c)

```

## Installation

This package requires ```scikit-learn```, ```numpy```, ```tensorflow```.

To install, you can run:

```
cd tfdnn
python setup.py install
```

## Currently supported features
It supports standard regularization with the *Frobenius norm* and also *Dropout* regularization.  
It allows the users to customize the Neural Network with the preferred activation function.    
Different weight initilizalization techniques are also implemented. 

The following modules are implemented: 
1. LogisticRegression
2. NeuralNetwork
3. DeepNeuralNetwork


### TODO
1. Support for sparse tensors.
2. Support for Tensorflow Datasets
3. Implement save and restore API.
4. Support for GPU computation
