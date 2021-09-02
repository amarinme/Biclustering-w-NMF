
import numpy as np

import NMF
import utils

# Lectura del fichero y conversion a Matriz
nameFile = "Input_Data/GDS3139_Micro.txt"


matrix_str = np.loadtxt(nameFile, dtype=str)
conditions = matrix_str[0][1:]  # guardo la cabecera

# convierto todos los numeros de la matriz de String a float
matrix_w_floats = utils.convert_numbers_to_float(matrix_str)
print("--------Caso A--------")
try:
# CASO A: Elimino la negatividad, si encuentro un elemento negativo elimino la fila

    non_negative_matrix_del = utils.delete_negative_rows(matrix_w_floats)
    genes_del = utils.obtain_labels(non_negative_matrix_del)
    non_negative_matrix_del = utils.without_labels(non_negative_matrix_del)

except:
    print("Matriz caso A vacia")

# CASO B: Elimino la negatividad, cuando encuentro un numero negativo lo cambio por 0, matengo la fila
print("--------Caso B--------")

genes = utils.obtain_labels(matrix_w_floats)
non_negative_matrix = utils.without_labels(matrix_w_floats)
non_negative_matrix = utils.change_negative_numbers(non_negative_matrix)


# GUARDANDO LOS RESULTADOS PARA CASO A
try:
    V1 = utils.add_labels(NMF.NMF(non_negative_matrix_del, 50, 5), genes_del, conditions)
    utils.write_in_csv_file(V1, "output_data/Result_NMF_casoA_Prueba8.csv")

    V2 = utils.add_labels(NMF.nsNMF(non_negative_matrix_del, 50, 5), genes_del, conditions)
    utils.write_in_csv_file(V2, "output_data/Result_nsNMF_casoA_Prueba8.csv")
except:
    print("No se puede realizar la escritura del caso A en el fichero para ", nameFile)

# GUARDANDO LOS RESULTADOS PARA CASO B

V3 = utils.add_labels(NMF.NMF(non_negative_matrix, 100, 20), genes, conditions)
utils.write_in_csv_file(V3, "output_data/Result_NMF_Prueba15_20k.csv")

V4 = utils.add_labels(NMF.nsNMF(non_negative_matrix, 100, 20), genes, conditions)
utils.write_in_csv_file(V4, "output_data/Result_nsNMF_Prueba15_20k.csv")




