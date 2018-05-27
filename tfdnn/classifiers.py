from sklearn.base import BaseEstimator, ClassifierMixin
from abc import abstractmethod
from .graph import LogisticRegressionGraph, \
    ShallowNeuralNetworkGraph, DeepNeuralNetworkGraph
from .utility import cross_entropy
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class BaseClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 dtype=tf.float32,
                 seed=1, shuffle=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        self.session = tf.Session(graph=self.graph)
        self.dtype = dtype
        self.shuffle = shuffle

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def score(self, X, y=None, sample_weight=None):
        pass


class LogisticRegression(BaseClassifier):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 dtype=tf.float32,
                 seed=1,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs={},
                 learning_rate=0.01, l2_w=0.01,
                 loss_function=cross_entropy,
                 init_std=0.01,
                 shuffle=True,
                 act_fun='sigmoid'):
        super(LogisticRegression, self).__init__(epochs=epochs,
                                                 batch_size=batch_size,
                                                 dtype=dtype,
                                                 seed=seed,
                                                 shuffle=shuffle)
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.learning_rate = learning_rate
        self.l2_w = l2_w
        self.init_std = init_std
        self.loss_function = loss_function
        self.act_fun = act_fun

        self.core = LogisticRegressionGraph(optimizer=self.optimizer,
                                            opt_kwargs=self.opt_kwargs,
                                            learning_rate=self.learning_rate,
                                            l2_w=self.l2_w, init_std=self.init_std,
                                            loss_function=self.loss_function,
                                            dtype=self.dtype,
                                            act_fun=self.act_fun)

    def fit(self, X, y):
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        with self.graph.as_default():
            self.core.define_graph()

        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size
        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars,
                             feed_dict={self.core.x: X[:batch_size]})

        idx = np.arange(n_samples)

        for _ in tqdm(range(self.epochs)):
            if self.shuffle:
                np.random.shuffle(idx)
            for ndx in range(0, n_samples, batch_size):
                batch_idx = idx[ndx: min(ndx + batch_size, n_samples)]
                batch_x = X[batch_idx]
                batch_y = y[batch_idx]
                self.session.run(self.core.trainer,
                                 feed_dict={self.core.x: batch_x,
                                            self.core.y: batch_y})

    def predict(self, X):
        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size

        idx = np.arange(n_samples)

        res = []
        for ndx in range(0, n_samples, batch_size):
            batch_idx = idx[ndx: min(ndx + batch_size, n_samples)]
            batch_x = X[batch_idx]
            batch_res = self.session.run(self.core.y_hat,
                                         feed_dict={self.core.x: batch_x})
            res.append(batch_res)
        return np.concatenate(res)

    def score(self, X, y=None, sample_weight=None):
        y_hat = (self.predict(X).reshape(-1, 1) > 0.5).astype(np.float)
        return accuracy_score(y_pred=y_hat, y_true=y)


class NeuralNetwork(BaseClassifier):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 dtype=tf.float32,
                 seed=1,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 learning_rate=0.01,
                 loss_function=cross_entropy,
                 init_std=0.01,
                 shuffle=True,
                 reg_hidden=0.01,
                 reg_output=0.01,
                 hidden_units=10,
                 keep_prob=None,
                 act_fun='sigmoid'):
        super(NeuralNetwork, self).__init__(epochs=epochs,
                                            batch_size=batch_size,
                                            dtype=dtype,
                                            seed=seed,
                                            shuffle=shuffle)
        if opt_kwargs is None:
            opt_kwargs = {}
        self.optimizer = optimizer
        self.opt_kwargs=opt_kwargs
        self.learning_rate = learning_rate
        self.reg_hidden = reg_hidden
        self.reg_output = reg_output
        self.init_std = init_std
        self.loss_function = loss_function
        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.act_fun = act_fun

        self.core = ShallowNeuralNetworkGraph(dtype=self.dtype,
                                              reg_hidden=self.reg_hidden,
                                              reg_output=self.reg_output,
                                              init_std=self.init_std,
                                              loss_function=self.loss_function,
                                              hidden_units=self.hidden_units,
                                              learning_rate=self.learning_rate,
                                              optimizer=self.optimizer,
                                              opt_kwargs=self.opt_kwargs,
                                              keep_prob=self.keep_prob,
                                              act_fun=self.act_fun)

    def fit(self, X, y):
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        with self.graph.as_default():
            self.core.define_graph()

        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size
        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars,
                             feed_dict={self.core.x: X[:batch_size]})

        idx = np.arange(n_samples)

        for _ in tqdm(range(self.epochs)):
            if self.shuffle:
                np.random.shuffle(idx)
            for ndx in range(0, n_samples, batch_size):
                batch_idx = idx[ndx: min(ndx + batch_size, n_samples)]
                batch_x = X[batch_idx]
                batch_y = y[batch_idx]
                self.session.run(self.core.trainer,
                                 feed_dict={self.core.x: batch_x,
                                            self.core.y: batch_y})

    def predict(self, X):
        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size

        idx = np.arange(n_samples)

        res = []
        for ndx in range(0, n_samples, batch_size):
            batch_idx = idx[ndx: min(ndx + batch_size, n_samples)]
            batch_x = X[batch_idx]
            batch_res = self.session.run(self.core.y_hat,
                                         feed_dict={self.core.x: batch_x})
            res.append(batch_res)
        return np.concatenate(res)

    def score(self, X, y=None, sample_weight=None):
        y_hat = self.predict(X)
        y = y.reshape(-1)
        y_hat = (y_hat.reshape(-1, 1) > 0.5).astype(np.float)
        return accuracy_score(y_pred=y_hat, y_true=y)


class DeepNeuralNetwork(BaseClassifier):

    def __init__(self,
                 epochs=100,
                 batch_size=-1,
                 dtype=tf.float32,
                 seed=1,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 learning_rate=0.01,
                 loss_function='cross_entropy',
                 init_std=0.01,
                 regularization=0.01,
                 shuffle=True,
                 hidden_units=None,
                 keep_prob=None,
                 act_fun='sigmoid',
                 momentum_batch_norm=None,
                 eps_batch_norm=10e-8):
        super(DeepNeuralNetwork, self).__init__(epochs=epochs,
                                                batch_size=batch_size,
                                                dtype=dtype,
                                                seed=seed,
                                                shuffle=shuffle)
        if opt_kwargs is None:
            opt_kwargs = {}
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.learning_rate = learning_rate
        self.init_std = init_std
        self.loss_function = loss_function
        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.act_fun = act_fun
        self.regularization = regularization
        self.eps_batch_norm = eps_batch_norm
        self.momentum_batch_norm = momentum_batch_norm

        self.core = DeepNeuralNetworkGraph(dtype=self.dtype,
                                           regularization=self.regularization,
                                           learning_rate=self.learning_rate,
                                           optimizer=self.optimizer,
                                           opt_kwargs=self.opt_kwargs,
                                           act_fun=self.act_fun,
                                           loss_function=self.loss_function,
                                           init_std=self.init_std,
                                           keep_prob=self.keep_prob,
                                           hidden_units=self.hidden_units,
                                           momentum_batch_norm=self.momentum_batch_norm,
                                           eps_batch_norm=self.eps_batch_norm)

    def fit(self, X, y):
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        self.core.set_params(**{'n_features': X.shape[1],
                                'n_classes': y.shape[1]})
        with self.graph.as_default():
            self.core.define_graph()

        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size
        if not self.session.run(tf.is_variable_initialized(
                self.core.global_step)):
            self.session.run(self.core.init_all_vars)

        idx = np.arange(n_samples)

        for _ in tqdm(range(self.epochs)):
            if self.shuffle:
                np.random.shuffle(idx)
            for ndx in range(0, n_samples, batch_size):
                batch_idx = idx[ndx: min(ndx + batch_size, n_samples)]
                batch_x = X[batch_idx]
                batch_y = y[batch_idx]
                self.session.run(self.core.trainer,
                                 feed_dict={self.core.x: batch_x,
                                            self.core.y: batch_y,
                                            self.core.is_training: True})

    def predict(self, X):
        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size == -1 else self.batch_size

        idx = np.arange(n_samples)

        res = []
        for ndx in range(0, n_samples, batch_size):
            batch_idx = idx[ndx: min(ndx + batch_size, n_samples)]
            batch_x = X[batch_idx]
            batch_res = self.session.run(self.core.y_hat,
                                         feed_dict={self.core.x: batch_x,
                                                    self.core.is_training: False})
            res.append(batch_res)
        return np.concatenate(res)

    def score(self, X, y=None, sample_weight=None):
        y_hat = self.predict(X)
        if len(y.shape) == 2:
            y_hat = np.argmax(y_hat, axis=1)
            y = np.argmax(y, axis=1)
        else:
            y = y.reshape(-1)
            y_hat = (y_hat.reshape(-1, 1) > 0.5).astype(np.float)
        return accuracy_score(y_pred=y_hat, y_true=y)