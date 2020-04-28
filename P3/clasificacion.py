# -*- coding: utf-8 -*-
"""
@author: Carlos Sánchez Muñoz
"""

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np
import matplotlib.pyplot as plt
import math

# Fijamos la semilla
np.random.seed(1)

#-------------------------------------------------------------------------------#
#-------------------------------- Clasificación --------------------------------#
#-------------------------------------------------------------------------------#

""" Funcion para leer los datos """
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []

	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 4:
				y.append(-1)
			else:
				y.append(1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

#---------------------------- Apartado 1 y 2 -----------------------------------#

"""Calcula el hiperplano que hace de clasificador binario.
Devuelve el vector de pesos y el número de iteraciones.
- datos: matriz de datos.
- labels: etiquetas.
- max_iters: número máximo de iteraciones.
- vini: valor inicial."""
def PLA_Pocket(datos, labels, max_iter, vini):
    w = vini.copy()
    w_best = w.copy()
    err_best = get_err(datos, labels, w_best)

    for it in range(1, max_iter + 1):
        w_last = w.copy()
        for dato, label in zip(datos, labels):
            if signo(w.dot(dato)) != label:
                w += label * dato

		# calculamos el error
        err = get_err(datos, labels, w)

		# Si mejoramos el error
        if err < err_best:
            w_best = w.copy()
            err_best = err

		# Si no hay cambios fin
        if np.all(w == w_last):
            return w_best

    return w_best


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n#################################")
	print("########  CLASIFICACIÓN  ########")
	print("#################################")

if __name__ == "__main__":
	main()
