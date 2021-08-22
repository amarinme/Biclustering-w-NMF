import utils
import math

import tensorflow as tf
import numpy as np

# Aqui es donde pondre el algoritmo que haga de NFC
# Mientras que lo estoy desarrollando lo estoy haciendo dentro del main con valores que yo le asigno

def NMF(non_negative_matrix, n_iter, H, W):

    V = np.array(non_negative_matrix)
    V_ = tf.constant(V)


    # m,n tamaño de mi matriz
    m, n = V.shape
    k = 10

    # Creo las matrices W y H (las inicializo aleatoriamente)
    W1, H1 = utils.random_initialization(m, n, k)

    #creo las variables en tensorflow
    H = tf.Variable(H1)
    W = tf.Variable(W1)
    W_Copy = tf.Variable(W1)

    # Multiplicative Update (MU ALGORITMO)

    # Mean sqrt (V.mean ()/rank) scale random uniform
    scale = 2 * math.sqrt(V.mean() / k)
    initializer = tf.random_uniform_initializer(maxval=scale)

    # WH = tf.matmul(W, H)

    # Realizamos las transformaciones en la matriz H

    Wt = tf.transpose(W)
    W_update = tf.matmul(Wt, V_) / tf.matmul(tf.matmul(Wt, W), H)

    #Me han salido algunos errores ya que al hacer operaciones algunas operaciones no eran validas
    # por lo que elimino los resultados Nan y los cambio por 0
    W_update = tf.select(tf.is_nan(W_update),tf.zeros_like(W_update),W_update)

    H_new = H * W_update
    H = H.assign(H_new)

    # Realizamos las mismas transformaciones pero esta veaz para W

    Ht = tf.transpose(H)

    H_update = tf.matmul(V_, Ht) / tf.matmul(W,tf.matmul(H, Ht))

    H_update = tf.select(tf.is_nan(H_update),tf.zeros_like(H_update),H_update)

    W_new = W * H_update
    W = W.assign(W_new)
