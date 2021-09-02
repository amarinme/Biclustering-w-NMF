import numpy as np
import tensorflow as tf
import pandas as pd


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
    lista = []

    for row in matrix:
        negative_exist = False

        for element in row[1:]:

            if element < .0:
                negative_exist = True
                break

        if not negative_exist:
            lista.append(row)


    end = len(lista)

    print("Número de filas negativas -> ", start - end)
    if end < 100:
        print("WARNING! Your Non Negative Matrix is very small, Total rows -> ", end)

    return lista

def change_negative_numbers(matrix):
    # Cada elemento negativo lo cambio por el numero 0
    count = 0
    count_elements = 0
    for row in matrix:
        for element in row:
            count_elements +=1
            if element < .0:
                i = matrix.index(row)
                j = row.index(element)
                matrix[i][j] = 0.0
                count +=1

    print("De ", count_elements, " en total -> ", count, " son negativos")
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
    for row in matrix:
        line = row[1:]
        final_matrix.append(line)

    return final_matrix


def random_initialization(m, n, k):
    # Crea dos matrices m y n de numeros aleatorios
    W = np.random.uniform(0, 1, (m, k))
    H = np.random.uniform(0, 1, (k, n))
    return W, H


def add_labels(matrix, genes, conditions):
    # Añade la cabecera con las condiciones y los genes
    fileName = "output_data/auxiliar.txt"
    np.savetxt(fileName, matrix)
    matrix = np.loadtxt(fileName, dtype=str)

    final = pd.DataFrame(matrix, index=genes, columns=conditions)
    return final

def write_in_csv_file(df, nameFile):
    # Escribimos el dataFrame en un fichero dado
    df.to_csv(nameFile)

def write_in_file(df, nameFile):
    # Escribimos el dataFrame en un fichero dado
    tfile = open(nameFile, 'w')
    tfile.write(df.to_string())
    tfile.close()

def empty_list(l):
    res = True
    if len(l) == 0:
        res = False

    return res


