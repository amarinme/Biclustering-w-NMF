import utils
import math

import tensorflow as tf
import numpy as np


# Aqui es donde pondre el algoritmo que haga de NFC
# Mientras que lo estoy desarrollando lo estoy haciendo dentro del main con valores que yo le asigno

def NMF(V, n_iter, k):
    try:
        V = np.array(V)
        V_ = tf.constant(V)

        # m,n tamaño de mi matriz
        m, n = V.shape


        # Creo las matrices W y H (las inicializo aleatoriamente)
        W1, H1 = utils.random_initialization(m, n, k)

        # creo las variables en tensorflow
        H = tf.Variable(H1)
        W = tf.Variable(W1)

        # Multiplicative Update (MU ALGORITMO)

        # Realizamos las transformaciones en la matriz H
        Wt = tf.transpose(W)
        W_update = tf.matmul(Wt, V_) / tf.matmul(tf.matmul(Wt, W), H)

        # Elimino los resultados Nan y los cambio por 0
        W_update = tf.where(tf.math.is_nan(W_update), tf.zeros_like(W_update), W_update)

        H_new = H * W_update
        H = H.assign(H_new)

        # Realizamos las mismas transformaciones pero esta veaz para W

        Ht = tf.transpose(H)

        H_update = tf.matmul(V_, Ht) / tf.matmul(W, tf.matmul(H, Ht))

        H_update = tf.where(tf.math.is_nan(H_update), tf.zeros_like(H_update), H_update)

        W_new = W * H_update
        W = W.assign(W_new)

        V = tf.matmul(W, H)

        if n_iter > 0:
            n_iter -= 1
            V = NMF(V, n_iter, k)

    except:
        print("Matriz invalida, probablemente vacia")

    return V

def nsNMF(V, n_iter, k):
    try:
        V = np.array(V)
        V_ = tf.constant(V)

        # m,n tamaño de mi matriz
        m, n = V.shape

        # Creo las matrices W y H (las inicializo aleatoriamente)
        W1, H1 = utils.random_initialization(m, n, k)

        # creo las variables en tensorflow
        H = tf.Variable(H1)
        W = tf.Variable(W1)

        # Multiplicative Update (MU ALGORITMO)

        # Realizamos las transformaciones en la matriz H
        Wt = tf.transpose(W)
        W_update = tf.matmul(Wt, V_) / tf.matmul(tf.matmul(Wt, W), H)

        # Elimino los resultados Nan y los cambio por 0
        W_update = tf.where(tf.math.is_nan(W_update), tf.zeros_like(W_update), W_update)

        H_new = H * W_update
        H = H.assign(H_new)

        # Realizamos las mismas transformaciones pero esta veaz para W

        Ht = tf.transpose(H)

        H_update = tf.matmul(V_, Ht) / tf.matmul(W, tf.matmul(H, Ht))

        H_update = tf.where(tf.math.is_nan(H_update), tf.zeros_like(H_update), H_update)

        W_new = W * H_update
        W = W.assign(W_new)

        S = smooth_matrix(k, 0.2)

        V = tf.matmul(tf.matmul(W, S), H)

        if n_iter > 0:
            n_iter -= 1
            V = NMF(V, n_iter, k)

    except:
        print("Matriz invalida, probablemente vacia")
    return V


def smooth_matrix(k, theta):
    # Creacion de la matriz de suavidad para el metodo nsNMF

    I = tf.eye(k)
    ones = tf.ones([k, k])
    S = ((1 - theta) * I) + (theta * tf.divide(ones, k))
    return np.array(S)

# Método alternativo en el que itero solo sobre las matrices W y H, no interviene V

# def NMF(non_negative_matrix, n_iter, k):
#     V = np.array(non_negative_matrix)
#     V_ = tf.constant(V)
#
#     # m,n tamaño de mi matriz
#     m, n = V.shape
#
#     # Creo las matrices W y H (las inicializo aleatoriamente)
#     W1, H1 = utils.random_initialization(m, n, k)
#
#     # creo las variables en tensorflow
#     H = tf.Variable(H1)
#     W = tf.Variable(W1)
#     W_Copy = tf.Variable(W1)
#
#     # Multiplicative Update (MU ALGORITMO)
#
#     # Mean sqrt (V.mean ()/rank) scale random uniform
#     scale = 2 * math.sqrt(V.mean() / k)
#     initializer = tf.random_uniform_initializer(maxval=scale)
#
#     # WH = tf.matmul(W, H)
#
#     for i in range(n_iter):
#         # Realizamos las transformaciones en la matriz H
#         Wt = tf.transpose(W)
#         W_update = tf.matmul(Wt, V_) / tf.matmul(tf.matmul(Wt, W), H)
#
#         # Elimino los resultados Nan y los cambio por 0
#         W_update = tf.where(tf.math.is_nan(W_update), tf.zeros_like(W_update), W_update)
#
#         H_new = H * W_update
#         H = H.assign(H_new)
#
#         # Realizamos las mismas transformaciones pero esta veaz para W
#
#         Ht = tf.transpose(H)
#
#         H_update = tf.matmul(V_, Ht) / tf.matmul(W, tf.matmul(H, Ht))
#
#         H_update = tf.where(tf.math.is_nan(H_update), tf.zeros_like(H_update), H_update)
#
#         W_new = W * H_update
#         W = W.assign(W_new)
#
#     return W, H
