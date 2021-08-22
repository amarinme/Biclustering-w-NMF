# tf.compat.v1.reset_default_graph()
import math

import tensorflow as tf
import numpy as np
import re

import NMF
import utils

# Lectura del fichero y conversion a Matriz
nameFile = "Input_Data/GDS3289_Micro.txt"

matrix_str = np.loadtxt(nameFile, dtype=str)
conditions = matrix_str[0]  # guardo la cabecera

# convierto todos los numeros de la matriz de String a float
matrix_w_floats = utils.convert_numbers_to_float(matrix_str)

# CASO A: Elimino la negatividad, si encuentro un elemento negativo elimino la fila

non_negative_matrix_del = utils.without_labels(utils.delete_negative_rows(matrix_w_floats))
genes_del = utils.obtain_labels(non_negative_matrix_del)

genes = utils.obtain_labels(matrix_w_floats)

# CASO B: Elimino la negatividad, cuando encuentro un numero negativo lo cambio por 0, matengo la fila

non_negative_matrix = utils.without_labels(matrix_w_floats)
non_negative_matrix = utils.change_negative_numbers(non_negative_matrix)

# GUARDANDO LOS RESULTADOS PARA CASO A

V1 = NMF.NMF(non_negative_matrix_del, 50, 5)
np.savetxt("output_data/Final_Result(NMF_del).txt", V1)

V2 = NMF.nsNMF(non_negative_matrix_del, 50, 5)
np.savetxt("output_data/Final_Result(nsNMF_del).txt", V2)

# V1 = NMF.NMF(non_negative_matrix_del, 50, 5)
# np.savetxt("output_data/Final_Result(NMF_del).csv", V1)
#
# V2 = NMF.nsNMF(non_negative_matrix_del, 50, 5)
# np.savetxt("output_data/Final_Result(nsNMF_del).csv", V2)

# GUARDANDO LOS RESULTADOS PARA CASO B

V1 = NMF.NMF(non_negative_matrix, 50, 5)
np.savetxt("output_data/Final_Result(NMF).txt", V1)

V2 = NMF.nsNMF(non_negative_matrix, 50, 5)
np.savetxt("output_data/Final_Result(nsNMF).txt", V2)

# V1 = NMF.NMF(non_negative_matrix, 50, 5)
# np.savetxt("output_data/Final_Result(NMF).csv", V1)
#
# V2 = NMF.nsNMF(non_negative_matrix, 50, 5)
# np.savetxt("output_data/Final_Result(nsNMF).csv", V2)



#DESCOMENTAR LO SIGUIENTE PARA LA PRUEBA DE AÑADIR LAS ETIQUETAS

# V1 = NMF.NMF(non_negative_matrix, 50, 5)
# V1 = utils.add_labels(V1, genes, conditions)
#
# np.savetxt("output_data/Final_Result(Prueba).txt", V1)
