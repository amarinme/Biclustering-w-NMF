import numpy as np
import tensorflow as tf
from numpy import dtype


def print_matrix(matrix):
    for m in matrix:
        print(m)


# los numeros de la matriz estan en tipo str por lo que con esta funcion los pasamos a tipo float

def convert_numbers_to_float(matrix):
    matrix_non_negative = []
    for linea in matrix[1:]:
        lista = [linea[0]]
        lista_float = [float(i) for i in linea[1:]]
        lista.extend(lista_float)
        matrix_non_negative.append(lista)

    return matrix_non_negative


def delete_negative_rows(matrix):
    # Eliminar las lineas con elementos negativos para cumplir la condicion de non Negatividad
    start = len(matrix)

    for row in matrix:
        for element in row[1:]:
            if element < .0:
                matrix.pop(matrix.index(row))
                break

    end = len(matrix)

    print("Number of negative rows -> ", start - end)
    if end < 100:
        print("WARNING! Your Non Negative Matrix is very small, Total rows -> ", end)

    return matrix


def change_negative_numbers(matrix):
    # Cada elemento negativo lo cambio por el numero 0
    count = 0
    for row in matrix:
        for element in row:
            if element < .0:
                i = matrix.index(row)
                j = row.index(element)
                matrix[i][j] = 0.0
                count +=1

    print("Se han localizado un total de " + str(count) + " elementos negativos")
    return matrix


def obtain_labels(matrix):
    # Guarda las etiquetas de cada fila y las devuelve en una lista
    labels = []
    for row in matrix:
        labels.append(row[0])
    return labels


def without_labels(matrix):
    # Metodo que elimina las etiquetas de cada linea
    final_matrix = []

    for row in matrix[1:]:
        line = row[1:]
        final_matrix.append(line)

    return final_matrix


def random_initialization(m, n, k):
    # Crea dos matrices m y n de numeros aleatorios
    W = np.random.uniform(0, 1, (m, k))
    H = np.random.uniform(0, 1, (k, n))
    return W, H


# def add_genes(matrix, genes):
#     # Metodo que elimina las etiquetas de cada lista
#     # Labels es una lista en la que guardo todas las etiquetas de cada fila
#     final_matrix = []
#     i = 0
#     for row in matrix[1:]:
#         line = row.insert(0, genes(i))
#         final_matrix.append(line)
#         i += 1
#     return final_matrix


def add_labels(matrix, genes, conditions):
    # Metodo que añade al array las etiquetas de los genes y de las matrices
    #Problema encontrado me indica que el dtype no es el mismo

    # Guardo la matriz en un fichero
    fileName = "output_data/Final_Result(Prueba).txt"
    np.savetxt(fileName, matrix)
    conditions = conditions[1:]
    conditions = tf.constant(conditions)

    # En aux es donde voy a crear mi nueva matrix con etiquetas, por lo que añado la primera file
    aux = [conditions]

    # Extraigo el fichero de nuevo con el formato que yo quiero
    matrix_str = np.loadtxt(fileName, dtype=str)
    matrix_str = tf.constant(matrix_str)

    aux.append(matrix_str)

    return aux


def empty_list(l):
    res = True
    if len(l) == 0:
        res = False

    return res

# def from_file_to_matrix(nameFile):
#     f = open(nameFile)
#     lines = f.readlines()
#
#     matrix = []
#     for l in lines:
#         s = l.split("\t")
#         matrix.append(s)
#
#     f.close()
#     return matrix
