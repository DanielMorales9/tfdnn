import tensorflow as tf
from abc import ABC, abstractmethod
from tfdnn.utility import cross_entropy, weight_init_coeff
from functools import reduce
import warnings

ACT_FUN = {'sigmoid': tf.sigmoid,
           'tanh': tf.tanh,
           'relu': tf.nn.relu,
           'leaky_relu': tf.nn.leaky_relu
           }

NUM_WEIGHT = { 'sigmoid': 1,
               'tanh': 1,
               'relu': 2,
               'leaky_relu': 2
               }

LOSS_FUN = {'cross_entropy': cross_entropy,
            'softmax_cross_entropy':
                lambda x, y: tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=y)}


class AbstractGraph(ABC):

    def __init__(self):
        self.global_step = None
        self.init_all_vars = None
        self.summary_op = None

    def define_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        with tf.name_scope('placeholders'):
            self.init_placeholders()
        with tf.name_scope('parameters'):
            self.init_params()
        with tf.name_scope('main_graph'):
            self.init_main_graph()
        with tf.name_scope('loss'):
            self.init_loss()
            self.init_regularization()
        with tf.name_scope('target'):
            self.init_target()
        with tf.name_scope('training'):
            self.init_trainer()
        self.init_all_vars = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def init_placeholders(self):
        pass

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def init_main_graph(self):
        pass

    @abstractmethod
    def init_loss(self):
        pass

    @abstractmethod
    def init_regularization(self):
        pass

    @abstractmethod
    def init_target(self):
        pass

    @abstractmethod
    def init_trainer(self):
        pass


class LogisticRegressionGraph(AbstractGraph):

    def __init__(self,
                 dtype=tf.float32,
                 l2_w=0.001,
                 learning_rate=0.01,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 act_fun='sigmoid',
                 loss_function=cross_entropy,
                 init_std=0.01):
        super(LogisticRegressionGraph, self).__init__()
        if opt_kwargs is None:
            opt_kwargs = {}
        assert dtype == tf.float32 or dtype == tf.float64, \
            'Dtype must be tf.float32 or tf.float64'
        self.dtype = dtype
        self.l2_w = l2_w
        self.learning_rate = learning_rate
        self.opt_kwargs = opt_kwargs
        self.optimizer = optimizer(learning_rate=self.learning_rate, **self.opt_kwargs)
        try:
            self.act_fun = ACT_FUN[act_fun]
            self.weight_init_numerator = NUM_WEIGHT[act_fun]
        except KeyError:
            raise ValueError(act_fun + ' is not supported. '
                                       'Please use one among '
                                       'the following supported '
                                       'activation functions: ' +
                             ' '.join(ACT_FUN.keys()))
        self.init_std = init_std
        self.loss_function = loss_function
        self.init_all_vars = None
        self.n_features = None
        self.summary_op = None
        self.lambda_w = None
        self.bias = None
        self.weights = None
        self.global_step = None
        self.l2_norm = None
        self.y_hat = None
        self.loss = None
        self.reduced_loss = None
        self.target = None
        self.checked_target = None
        self.trainer = None
        self.init_all_vars = None
        self.x = None
        self.y = None

    def init_placeholders(self):
        self.x = tf.placeholder(self.dtype, name='x')
        self.y = tf.placeholder(self.dtype, shape=[None, 1], name='y')

    def init_params(self):
        self.n_features = tf.shape(self.x)[1]
        self.lambda_w = tf.constant(self.l2_w, dtype=self.dtype, name='lambda_w')
        tf_shape = tf.stack([self.n_features, 1])
        coeff = weight_init_coeff(self.weight_init_numerator, self.dtype, self.n_features)
        rnd_weights = tf.multiply(coeff,
                                  tf.random_normal(tf_shape, dtype=self.dtype))
        self.weights = tf.verify_tensor_all_finite(
            tf.Variable(rnd_weights,
                        trainable=True,
                        validate_shape=False,
                        name='weights'),
            'NaN or Inf in weights')
        self.bias = tf.verify_tensor_all_finite(
            tf.Variable(self.init_std,
                        trainable=True,
                        name='bias'),
            'NaN or Inf in bias')
        tf.summary.scalar('bias', self.bias)

    def init_trainer(self):
        self.trainer = self.optimizer.minimize(self.checked_target,
                                               global_step=self.global_step)

    def init_regularization(self):
        norm = tf.norm(self.weights, ord=2, keepdims=True)
        two_m = self.lambda_w
        if self.dtype == tf.float32:
            two_m /= tf.to_float(2 * tf.shape(self.x)[0])
        else:
            two_m /= tf.to_double(2 * tf.shape(self.x)[0])
        self.l2_norm = tf.multiply(two_m, norm)

    def init_target(self):
        self.target = self.l2_norm + self.reduced_loss
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    def init_main_graph(self):
        z = self.x @ self.weights + self.bias
        self.y_hat = self.act_fun(z)

    def init_loss(self):
        self.loss = self.loss_function(self.y, self.y_hat)
        self.reduced_loss = tf.reduce_mean(self.loss)


class ShallowNeuralNetworkGraph(AbstractGraph):

    def __init__(self,
                 dtype=tf.float32,
                 reg_hidden=0.001,
                 reg_output=0.001,
                 learning_rate=0.01,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 act_fun='sigmoid',
                 loss_function=cross_entropy,
                 init_std=0.01,
                 hidden_units=10,
                 keep_prob=None):
        super(ShallowNeuralNetworkGraph, self).__init__()
        if opt_kwargs is None:
            opt_kwargs = {}
        assert dtype == tf.float32 or dtype == tf.float64, \
            'Dtype must be tf.float32 or tf.float64'
        self.dtype = dtype
        self.reg_hidden = reg_hidden
        self.reg_output = reg_output
        self.learning_rate = learning_rate
        self.opt_kwargs = opt_kwargs
        self.optimizer = optimizer(learning_rate=self.learning_rate, **self.opt_kwargs)
        try:
            self.act_fun = ACT_FUN[act_fun]
            self.weight_init_numerator = NUM_WEIGHT[act_fun]
        except KeyError:
            raise ValueError(act_fun + ' is not supported. '
                                       'Please use one among '
                                       'the following supported '
                                       'activation functions: ' +
                             ' '.join(ACT_FUN.keys()))

        self.init_std = init_std
        self.loss_function = loss_function
        self.hidden_units = hidden_units
        self.has_drop_out = keep_prob is not None
        self.keep_prob = keep_prob
        self.norm = None
        self.reduced_loss = None
        self.output_layer = None
        self.n_features = None
        self.lambda_hidden = None
        self.lambda_output = None
        self.hidden_layer = None
        self.hidden_bias = None
        self.output_bias = None
        self.target = None
        self.checked_target = None
        self.trainer = None
        self.y_hat = None
        self.loss = None
        self.x = None
        self.y = None

    def init_placeholders(self):
        self.x = tf.placeholder(self.dtype, name='x')
        self.y = tf.placeholder(self.dtype, shape=[None, 1], name='y')

    def init_params(self):
        self.n_features = tf.shape(self.x)[1]
        self.lambda_hidden = tf.constant(self.reg_hidden,
                                         dtype=self.dtype,
                                         name='lambda_hidden_params')
        self.lambda_output = tf.constant(self.reg_output,
                                         dtype=self.dtype,
                                         name='lambda_output_params')

        hidden_layer_shape = tf.stack([self.n_features, self.hidden_units])

        coeff = weight_init_coeff(self.weight_init_numerator, self.dtype, self.n_features)
        rnd_output_layers = tf.multiply(coeff,
                                        tf.random_normal(hidden_layer_shape, dtype=self.dtype))
        self.hidden_layer = tf.verify_tensor_all_finite(
            tf.Variable(rnd_output_layers,
                        trainable=True,
                        validate_shape=False,
                        name='weights'),
            'NaN or Inf in weights')

        self.hidden_bias = tf.verify_tensor_all_finite(
            tf.Variable(self.init_std,
                        trainable=True,
                        name='bias'),
            'NaN or Inf in bias')
        output_layer_shape = tf.stack([self.hidden_units, 1])
        coeff = weight_init_coeff(self.weight_init_numerator, self.dtype,
                                  tf.convert_to_tensor(self.hidden_units))
        rnd_output_layers = tf.multiply(coeff,
                                        tf.random_normal(output_layer_shape, dtype=self.dtype))
        self.output_layer = tf.verify_tensor_all_finite(
            tf.Variable(rnd_output_layers,
                        trainable=True,
                        validate_shape=False,
                        name='weights'),
            'NaN or Inf in weights')

        self.output_bias = tf.verify_tensor_all_finite(
            tf.Variable(self.init_std,
                        trainable=True,
                        name='bias'),
            'NaN or Inf in bias')

        tf.summary.scalar('hidden_bias', self.hidden_bias)
        tf.summary.scalar('output_bias', self.output_bias)

    def init_main_graph(self):
        z1 = self.x @ self.hidden_layer + self.hidden_bias
        a1 = self.act_fun(z1)

        if self.has_drop_out:
            rnd = tf.random_normal(tf.shape(a1), dtype=self.dtype)
            if self.dtype == tf.float32:
                d1 = tf.to_float(rnd < self.keep_prob)
            else:
                d1 = tf.to_double(rnd < self.keep_prob)
            a1 = a1 * d1

        z2 = a1 @ self.output_layer + self.output_bias
        a2 = self.act_fun(z2)
        self.y_hat = a2

    def init_loss(self):
        self.loss = self.loss_function(self.y, self.y_hat)
        self.reduced_loss = tf.reduce_mean(self.loss)

    def init_regularization(self):
        hidden_norm = tf.reduce_sum(tf.pow(self.hidden_layer, 2))
        hidden_norm = tf.multiply(self.lambda_hidden, hidden_norm)
        output_norm = tf.reduce_sum(tf.pow(self.output_layer, 2))
        output_norm = tf.multiply(self.lambda_output, output_norm)
        two_m = 1 / (2 * tf.shape(self.x)[0])
        if self.dtype == tf.float32:
            two_m = tf.to_float(two_m)
        else:
            two_m = tf.to_double(two_m)
        self.norm = tf.multiply(two_m, tf.add(hidden_norm, output_norm))

    def init_target(self):
        self.target = self.norm + self.reduced_loss
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    def init_trainer(self):
        self.trainer = self.optimizer.minimize(self.checked_target,
                                               global_step=self.global_step)


class DeepNeuralNetworkGraph(AbstractGraph):

    def __init__(self,
                 dtype=tf.float32,
                 regularization=0.001,
                 learning_rate=0.01,
                 optimizer=tf.train.AdamOptimizer,
                 opt_kwargs=None,
                 act_fun='sigmoid',
                 loss_function='cross_entropy',
                 init_std=0.01,
                 momentum_batch_norm=0.9,
                 eps_batch_norm=10e-8,
                 hidden_units=None,
                 keep_prob=None):
        super(DeepNeuralNetworkGraph, self).__init__()
        if opt_kwargs is None:
            opt_kwargs = {}
        assert dtype == tf.float32 or dtype == tf.float64, \
            'Dtype must be tf.float32 or tf.float64'
        if hidden_units is None:
            hidden_units = [10, 10]
        assert len(hidden_units) >= 2, \
            'Number of hidden layers must be at least two, ' \
            'otherwise use ShallowNeuralNetworkGraph'
        if type(keep_prob) is list:
            assert len(hidden_units) == len(keep_prob), \
                'hidden_units and keep_prob must be of equal length'

        self.dtype = dtype
        self.eps_batch_norm = eps_batch_norm
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.opt_kwargs = opt_kwargs
        self.optimizer = optimizer(learning_rate=self.learning_rate, **self.opt_kwargs)

        try:
            self.act_fun = ACT_FUN[act_fun]
            self.weight_init_numerator = NUM_WEIGHT[act_fun]
        except KeyError:
            raise ValueError(act_fun + ' is not supported. '
                                       'Please use one among '
                                       'the following supported '
                                       'activation functions: ' +
                             ' '.join(ACT_FUN.keys()))

        self.hidden_units = hidden_units
        self.keep_prob = keep_prob
        self.n_hidden_layers = len(self.hidden_units)
        self.n_layers = self.n_hidden_layers + 1
        if self.keep_prob is not None and type(self.keep_prob) is not list:
            self.keep_prob = [self.keep_prob] * self.n_hidden_layers

        self.init_std = init_std
        loss = LOSS_FUN[loss_function]
        if loss_function == 'cross_entropy':
            self.loss_function = lambda y, y_hat: loss(y, self.act_fun(y_hat))
        else:
            self.loss_function = loss
        self.has_drop_out = keep_prob is not None
        self.momentum_batch_norm = momentum_batch_norm
        self.has_batch_norm = self.momentum_batch_norm is not None
        self.layers = []
        self.bias = []
        self.n_classes = None
        self.loss = None
        self.reduced_loss = None
        self.norm = None
        self.target = None
        self.trainer = None
        self.checked_target = None
        self.n_features = None
        self.lambda_reg = None
        self.y_hat = None
        self.is_training = None
        self.x = None
        self.y = None

        if self.has_batch_norm and self.has_drop_out and self.regularization != 0.0:
            warnings.warn('You can use Batch-Norm and Dropout alternatively to L2-regularization')

    def init_placeholders(self):
        self.x = tf.placeholder(self.dtype, shape=[None, self.n_features], name='x')
        self.y = tf.placeholder(self.dtype, shape=[None, self.n_classes], name='y')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='train')

    def init_params(self):
        self.lambda_reg = tf.constant(self.regularization,
                                      dtype=self.dtype,
                                      name='lambda_regularization')
        hid_units = [self.n_features]
        hid_units.extend(self.hidden_units)
        hid_units.append(self.n_classes)

        for i in range(self.n_layers):
            shape = hid_units[i:i+2]
            rnd_layer = tf.random_normal(shape,
                                         stddev=self.init_std,
                                         dtype=self.dtype)
            ith_layer = tf.verify_tensor_all_finite(
                tf.Variable(rnd_layer,
                            trainable=True,
                            validate_shape=i < self.n_hidden_layers,
                            name='layer_'+str(i)),
                msg='NaN or Inf in layer_'+str(i))

            if not self.has_batch_norm or i >= self.n_hidden_layers:
                ith_bias = tf.verify_tensor_all_finite(
                    tf.Variable(tf.convert_to_tensor(self.init_std),
                                trainable=True,
                                name='bias_'+str(i)),
                    msg='NaN or Inf in bias_' + str(i))
                self.bias.append(ith_bias)
            self.layers.append(ith_layer)

    def init_main_graph(self):
        a = self.x
        m = tf.shape(self.x)[0]

        if self.dtype is tf.float32:
            m = tf.to_float(m)
        else:
            m = tf.to_double(m)

        lambda_reg = self.lambda_reg

        def regularizer(x):
            return (lambda_reg / (2 * m)) * tf.reduce_sum(tf.square(x))

        for i in range(self.n_hidden_layers):
            z = a @ self.layers[i]
            if not self.has_batch_norm:
                z += self.bias[i]

            a1 = self.act_fun(z)
            if self.has_batch_norm:
                a1 = tf.layers.batch_normalization(a1,
                                                   momentum=self.momentum_batch_norm,
                                                   epsilon=self.eps_batch_norm,
                                                   training=self.is_training,
                                                   beta_regularizer=regularizer,
                                                   gamma_regularizer=regularizer)
            if self.has_drop_out:
                a1 = tf.layers.dropout(a1, rate=self.keep_prob[i],
                                       training=self.is_training)
            a = a1

        if self.has_batch_norm:
            z = a @ self.layers[self.n_hidden_layers] + self.bias[0]
        else:
            z = a @ self.layers[self.n_hidden_layers] + self.bias[self.n_hidden_layers]

        self.y_hat = z

    def init_loss(self):
        self.loss = self.loss_function(self.y, self.y_hat)
        self.reduced_loss = tf.reduce_mean(self.loss)

    def init_regularization(self):
        norm = [tf.reduce_sum(tf.pow(x, 2)) for x in self.layers]
        norm = reduce(lambda x1, x2: x1 + x2, norm)
        two_m = 2 * tf.shape(self.x)[0]
        if self.dtype == tf.float32:
            two_m = tf.to_float(two_m)
        else:
            two_m = tf.to_double(two_m)

        self.norm = tf.multiply(self.lambda_reg / two_m, norm)

    def init_target(self):
        self.target = self.norm + self.reduced_loss
        self.checked_target = tf.verify_tensor_all_finite(
            self.target,
            msg='NaN or Inf in target value',
            name='target')
        tf.summary.scalar('target', self.checked_target)

    def init_trainer(self):
        self.trainer = self.optimizer.minimize(self.checked_target,
                                               global_step=self.global_step)


