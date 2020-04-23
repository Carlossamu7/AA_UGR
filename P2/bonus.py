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
#------------------------------------ Bonus ------------------------------------#
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

""" Calcula la tasa de error
- datos: matriz de datos.
- labels: etiquetas.
- fun: función clasificadora."""
def get_err(datos, labels, w):
	aciertos = labels*datos.dot(w)
	return 1-(len(aciertos[aciertos >= 0])/len(labels))

""" Calcula w usando el algoritmo de la pseudoinversa """
def pseudoinverse(x, y):
	x_pinv = np.linalg.inv(x.T.dot(x)).dot(x.T)	# (x^T * x)^-1 * x^T
	return np.dot(x_pinv, y)					# w = pseudoinversa(x) * y

""" Función signo de un valor real. El 0 tiene asignado signo positivo.
- x: valor del que saber el signo."""
def signo(x):
	if x >= 0:
		return 1
	return -1

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

""" Calcula y devuelve la cota de Eout.
- ein: error estimado.
- N: tamaño de la muestra.
- delta: tolerancia."""
def cotaErr(ein, N, delta):
	return ein + np.sqrt(1/(2*N)*((3*64+1)*np.log(2) - np.log(delta)))

""" Función que ejecuta todo el apartado 1 """
def apartado1y2():
	print ("\n###  Apartado 1  ###\n")
	print("Leyendo los ficheros de datos de train y test.")
	# Lectura de los datos de entrenamiento
	x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
	# Lectura de los datos para el test
	x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

	print ("\n###  Apartado 2  ###\n")

	# Calculamos los vectores de pesos
	print("Calculando los vectores de pesos con un modelo de regresión lineal.")
	w_pin = pseudoinverse(x, y)
	print("Aplicando mejora de PLA-Pocket.\n")
	w_pla = PLA_Pocket(x, y, 1000, w_pin.reshape(-1,))

	# Representamos las rectas obtenidas para train
	print("a) Gráfico de los datos de entrenamiento con la función estimada.")
	plt.scatter(x[y == -1][:, 1], x[y == -1][:, 2], label="Etiqueta -1")
	plt.scatter(x[y == 1][:, 1], x[y == 1][:, 2], c="orange", label="Etiqueta 1")
	points = np.array([np.min(x[:, 1]), np.max(x[:, 1])])
	plt.plot(points, (-w_pin[1]*points - w_pin[0])/w_pin[2], c="red", label="Pseudoinversa")
	plt.plot(points, (-w_pla[1]*points - w_pla[0])/w_pla[2], c="green", label="PLA-Pocket")
	plt.legend()
	plt.title("Regresión sobre dígitos manuscritos (train)")
	plt.gcf().canvas.set_window_title('Bonus')
	plt.show()

	# Representamos las rectas obtenidas para test
	print("   Gráfico de los datos de test con la función estimada.")
	plt.scatter(x_test[y_test == -1][:, 1], x_test[y_test == -1][:, 2], label="Etiqueta -1")
	plt.scatter(x_test[y_test == 1][:, 1], x_test[y_test == 1][:, 2], c="orange", label="Etiqueta 1")
	points = np.array([np.min(x_test[:, 1]), np.max(x_test[:, 1])])
	plt.plot(points, (-w_pin[1]*points - w_pin[0])/w_pin[2], c="red", label="Pseudoinversa")
	plt.plot(points, (-w_pla[1]*points - w_pla[0])/w_pla[2], c="green", label="PLA-Pocket")
	plt.legend()
	plt.title("Regresión sobre dígitos manuscritos (test)")
	plt.gcf().canvas.set_window_title('Bonus')
	plt.show()

	# Imprimimos el cálculo de los errores
	print("\nb) Calcular Ein y Etest.\n")
	print("   Ein   para Pseudoinversa: ", get_err(x, y, w_pin))
	print("   Etest para Pseudoinversa: ", get_err(x_test, y_test, w_pin))
	e_in = get_err(x, y, w_pla)
	e_test = get_err(x_test, y_test, w_pla)
	print("\n   Ein   para PLA-Pocket   : ", e_in)
	print("   Etest para PLA-Pocket   : ", e_test)
	input("\n--- Pulsar tecla para continuar ---\n")

	# Cálculo de las cotas de los errores
	delta = 0.05
	print("\nc) Cota de Eout.\n")
	print("   Usando Ein  : ", cotaErr(e_in, len(x), delta))
	print("   Usando Etest: ", cotaErr(e_test, len(x_test), delta))
	input("\n--- Pulsar tecla para continuar ---\n")

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n#############################")
	print("##########  BONUS  ##########")
	print("#############################")
	apartado1y2()

if __name__ == "__main__":
	main()
