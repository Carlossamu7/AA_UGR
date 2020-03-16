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

np.random.seed(1)	# Fijamos la semilla

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------ Apartado 1 -------------------------------------#

""" Funcion para leer los datos """
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []

	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

""" Funcion para calcular el error """
def Err(x,y,w):
	return (np.linalg.norm(x.dot(w) - y)**2) / len(x)

""" Derivada del error """
def dErr(x, y, w):
  return 2/len(x)*(x.T.dot(x.dot(w) - y))

""" Gradiente Descendente Estocastico.
- x: datos en coordenadas homogéneas.
- y: etiquetas asociadas {-1,1}.
- lr: tasa de aprendizaje.
- max_iters: número máximo de iteraciones.
- tam_minibatch: tamaño del minibatch. """
def sgd(x, y, lr, max_iters, tam_minibatch):
	w = np.zeros(3)
	it = 0
	ind_set = np.random.permutation(np.arange(len(x)))	 # conjunto de índices
	begin = 0					# Comienzo de la muestra

	while it < max_iters:
		if  begin > len(x):		# Nueva época
			begin = 0
			ind_set = np.random.permutation(ind_set)

		minibatch = ind_set[begin:begin + tam_minibatch] # Escogemos el minibatch
		w = w - lr*dErr(x[minibatch, :], y[minibatch], w)
		it += 1					# Acualizo iteraciones
		begin += tam_minibatch	# Cambio minibatch
	return w

""" Calcula w usando el algoritmo de la pseudoinversa """
def pseudoinverse(x, y):
	x_pinv = np.linalg.inv(x.T.dot(x)).dot(x.T)	# (X^T * X)^-1 * X^T
	return np.dot(x_pinv, y)					# w = pseudoinversa(x) * y

""" Función que ejecuta todo el apartado 1 """
def apartado1():
	print ("\n### Apartado 1 ###\n")
	# Lectura de los datos de entrenamiento
	x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
	# Lectura de los datos para el test
	x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

	# Gradiente descendente estocástico
	w = sgd(x, y, 0.01, 10000, 32)
	print ("Bondad del resultado para grad. descendente estocástico:")
	print ("Ein: ", Err(x, y, w))
	print ("Eout: ", Err(x_test, y_test, w))

	# Algoritmo Pseudoinversa
	w = pseudoinverse(x, y)
	print ("\nBondad del resultado para el algoritmo de la pseudoinversa:")
	print ("Ein: ", Err(x, y, w))
	print ("Eout: ", Err(x_test, y_test, w))

	input("--- Pulsar tecla para continuar ---\n")

#------------------------------ Apartado 2 -------------------------------------#

""" Simula datos en un cuadrado [-size,size]x[-size,size] """
def simula_unif(N, d, size):
	return np.random.uniform(-size, size, (N, d))

""" Calcula la función f dada sobre dos vectores x1 y x2.
	Introduce ruido al 10% de los datos (cambia las etiquetas). """
def f(x1, x2):
	# Calulamos la predicción de todos los puntos 2D
	res = np.sign((x1 - 0.2)**2 + x2**2 - 0.6)
	# Introducimos ruido en el 10%
	ind = np.random.choice(len(res), size=int(len(res)/10), replace=False)
	for i in ind:
		res[i] = -res[i]
	return res

""" Experemiento a ejecutar 1000 veces """
def experiment():
	x = np.hstack((np.ones((1000, 1)), simula_unif(1000, 2, 1)))
	y = f(x[:,0],x[:,1])
	x_test = np.hstack((np.ones((1000, 1)), simula_unif(1000, 2, 1)))
	y_test = f(x[:,0],x[:,1])
	w = sgd(x, y, 0.01, 1000, 32)
	Ein = Err(x, y, w)
	Eout = Err(x_test, y_test, w)
	return np.array([Ein, Eout])

""" Función que ejecuta todo el apartado 2 """
def apartado2():
	# EXPERIMENTO
	# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]
	print ("\n### Apartado 2 ###\n")
	print ("a) Muestra N = 1000, cuadrado [-1,1]x[-1,1]")
	x = simula_unif(1000, 2, 1)
	plt.scatter(x[:,0], x[:,1])
	plt.show()

	# b) Mapa de etiquetas usando la función f y con un 10% de ruido
	print ("\nb) Mapa de etiquetas")
	y = f(x[:,0],x[:,1])
	plt.scatter(x[y==1][:,0], x[y==1][:,1], label="Etiqueta 1")
	plt.scatter(x[y==-1][:,0], x[y==-1][:,1], c='orange', label="Etiqueta -1")
	plt.legend()
	plt.show()

	# c) Usando (1, x1, x2) ajustar un modelo de regresion lineal al conjunto de
	# datos y estimar los pesos w. Estimar el error de ajuste Ein usando SGD.
	x_ones = np.hstack((np.ones((1000, 1)), x))	# columna de 1s
	w = sgd(x_ones, y, 0.01, 1000, 32)
	print("\nc) Bondad del resultado para SGD:")
	print("   Ein:  {}".format(Err(x_ones, y, w)))
	input("--- Pulsar tecla para continuar ---")

	# d) Ejecutar el experimento 1000 veces
	print ("\nd) Errores Ein y Eout medios tras 1000reps del experimento:")
	N = 1000; errs = np.array([0.,0.])
	for _ in range(N):
		errs = errs + experiment()
	Ein_media, Eout_media = errs/N
	print ("   Ein media: ", Ein_media)
	print ("   Eout media: ", Eout_media)
	input("--- Pulsar tecla para continuar ---\n")

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n############################################")
	print("###  2 EJERCICIO SOBRE REGRESIÓN LINEAL  ###")
	print("############################################")

	apartado1()
	apartado2()

if __name__ == "__main__":
	main()
