import tensorflow as tf


def cross_entropy(y, yhat):
    loss = - y * tf.log(yhat) - (1 - y) * tf.log(1 - yhat)
    return loss


def weight_init_coeff(num, dtype, rows):
    if dtype == tf.float32:
        rows = tf.to_float(rows)
    elif dtype == tf.float64:
        rows = tf.to_double(rows)
    else:
        raise ValueError('Dtype must be tf.float32 or tf.float64')
    weight_coeff = tf.sqrt(num / rows)

    return weight_coeff
