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

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------ Apartado 1 -------------------------------------#


# Funcion para leer los datos
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

# Funcion para calcular el error
def Err(x,y,w):
	return

# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):
	return w

# Algoritmo pseudoinversa
def pseudoinverse(x, y):

	return w

# Lectura de los datos de entrenamiento
x, y = readData()
# Lectura de los datos para el test
x_test, y_test = readData()

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err())
print ("Eout: ", Err())

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err())
print ("Eout: ", Err())


#------------------------------ Apartado 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return

# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')




# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")

########################
#####     MAIN     #####
########################

""" Programa principal. """
#def main():
#	print("\n############################################")
#	print("###  2 EJERCICIO SOBRE REGRESIÓN LINEAL  ###")
#	print("############################################")
#	apartado1()
#	apartado2()

#if __name__ == "__main__":
#	main()
