# -*- coding: utf-8 -*-
"""
@author: Carlos Sánchez Muñoz
"""

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

#-------------------------------------------------------------------------------#
#----------------------------- Modelos Lineales --------------------------------#
#-------------------------------------------------------------------------------#

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_recta(intervalo):
	points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
	x1 = points[0,0]
	x2 = points[1,0]
	y1 = points[0,1]
	y2 = points[1,1]
	# y = a*x + b
	a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
	b = y1 - a*x1       # Calculo del termino independiente.

	return a, b

""" Función signo de un valor real. El 0 tiene asignado signo positivo.
- x: valor del que saber el signo."""
def signo(x):
	if x >= 0:
		return 1
	return -1

""" Función clasificadora mediante una recta
- x: primera variable de la función.
- y: segunda variable de la función.
- a: pendiente de la recta.
- b: ordenada en el origen de la recta."""
def f(x, y, a, b):
	return y - a*x - b

#------------------------------ Apartado 1 -------------------------------------#

""" Calcula el hiperplano solución a un problema de clasificación binaria.
Devuelve el vector de pesos y el número de iteraciones.
- datos: matriz de datos,
- labels: etiquetas,
- max_iter: número máximo de iteraciones
- vini: Valor inicial
"""
def ajusta_PLA(datos, labels, max_iter, vini):
	w = vini.copy()

	for it in range(1, max_iter + 1):
		w_old = w.copy()

		for dato, label in zip(datos, labels):
			if signo(w.dot(dato)) != label:
				w += label*dato

		if np.all(w == w_old):  # No hay cambios
			return w, it

	return w, it

""" Calcula el porcentaje de puntos bien clasificados
- datos: datos.
- labels: etiquetas.
- fun: función clasificadora."""
def get_porc(datos, labels, w):
	signos = labels*datos.dot(w)
	return 100*len(signos[signos >= 0])/len(labels)

""" Ejecuta ajusta_PLA() con vini un vector de ceros y luego 10 aleatorios
- datos: matriz de datos,
- labels: etiquetas,
- max_iter: número máximo de iteraciones
"""
def ejecuta_PLA(datos, labels, max_iters):
	print("   Vector inicial cero")
	w, it = ajusta_PLA(datos, labels, max_iters, np.zeros(3))
	print("   Num. iteraciones: {}".format(it))
	print("   Acierto: {}%".format(get_porc(datos, labels, w)))

	print("\n   Diez vectores iniciales aleatorios")
	iters = np.empty((10, ))
	percs = np.empty((10, ))
	for i in range(10):
		w, it = ajusta_PLA(datos, labels, max_iters, np.random.rand(3))
		iters[i] = it
		percs[i] = get_porc(datos, labels, w)
	print("   N. iteraciones: {}".format(np.mean(iters)))
	print("   Aciertos: {}%".format(np.mean(percs)))

""" Función que ejecuta todo el apartado 1 """
def apartado1():
	print ("\n###  Apartado 1  ###\n")

	print("a) Ejecutar PLA con los datos del ejercicio 1.2a).\n")
	N = 50
	a, b = simula_recta([-50, 50])
	x = np.hstack((np.ones((N, 1)), simula_unif(N, 2, [-50, 50])))
	y = np.empty((N, ))
	for i in range(N):
		y[i] = signo(f(x[i,1], x[i,2], a, b))
	ejecuta_PLA(x, y, 1000)

	print("\nb) Ejecutar PLA con los datos del ejercicio 1.2b).\n")
	y_noise = np.copy(y)	# Introducimos ruido en el 10%
	ind = np.random.choice(N, size=int(N/10), replace=False)
	for i in ind:
		y_noise[i] = -y[i]
	ejecuta_PLA(x, y_noise, 1000)

	input("\n--- Pulsar tecla para continuar ---")

#------------------------------ Apartado 2 -------------------------------------#

def grad_RL(dato, label, w):
	return -label*dato/(1 + np.exp(label*w.dot(dato)))

"""Implementa el algoritmo de regresión logística
mediante SGD con tamaño de batch 1.
Argumentos posicionales:
- datos: datos y
- labels: etiquetas.
Devuelve: Vector w que define el clasificador.
"""
def sgd_RL(datos, labels, eta=0.01):
	N, dim = datos.shape
	w = np.zeros(dim)
	ha_cambiado = True  # Si ha variado en la época actual
	idxs = np.arange(N)  # vector de índices

	while ha_cambiado:
		w_old = w.copy()
		idxs = np.random.permutation(idxs)
		for idx in idxs:
			w += -eta*grad_RL(datos[idx], labels[idx], w)
		ha_cambiado = np.linalg.norm(w - w_old) > 0.01

	return w

"""Calcula el error de un clasificador logístico para una muestra de datos.
Argumentos opcionales:
- w: Vector de pesos del clasificador logístico,
- x: datos en coordenadas homogéneas,
- y: etiquetas
Devuelve:
- El error logístico"""
def Err_RL(datos, labels, w):
	return np.mean(np.log(1 + np.exp(-labels*datos.dot(w))))

""" Función que ejecuta todo el apartado 2 """
def apartado2():
	print ("\n###  Apartado 2  ###\n")

	print("a) Implementar RL con SGD. Mostramos gráfica con el resultado.")
	# Calculamos datos y labels
	N = 100; intervalo = [0, 2]
	a, b = simula_recta([0, 2])
	datos = np.hstack((np.ones((N, 1)), simula_unif(N, 2, intervalo)))
	labels = np.empty((N, ))
	for i in range(N):
		labels[i] = signo(f(datos[i, 1], datos[i, 2], a, b))

	# Calculamos el vector de pesos usando RL+SGD
	w = sgd_RL(datos, labels)

	# Representamos la recta obtenida
	plt.scatter(datos[labels == -1][:, 1], datos[labels == -1][:, 2], label="Etiqueta -1")
	plt.scatter(datos[labels == 1][:, 1], datos[labels == 1][:, 2], c="orange", label="Etiqueta 1")
	points = np.array([np.min(datos[:, 1]), np.max(datos[:, 1])])
	plt.plot(points, (-w[1]*points - w[0])/w[2], c="red", label="Recta RL")
	plt.legend()
	plt.title("Regresión Logística con SGD")
	plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2a)')
	plt.show()

	print("\nb) Encontrar solución g y estimar Eout con nuevas muestras.\n")
	# Calculamos datos y labels de test (uso el mismo N>999)
	datos_test = np.hstack((np.ones((N, 1)), simula_unif(N, 2, intervalo)))
	labels_test = np.empty((N, ))
	for i in range(N):
		labels_test[i] = signo(f(datos_test[i, 1], datos_test[i, 2], a, b))

	# Representamos la recta y el conjunto de test
	plt.scatter(datos_test[labels_test == -1][:, 1], datos_test[labels_test == -1][:, 2], label="Etiqueta -1")
	plt.scatter(datos_test[labels_test == 1][:, 1], datos_test[labels_test == 1][:, 2], c="orange", label="Etiqueta 1")
	points = np.array([np.min(datos_test[:, 1]), np.max(datos_test[:, 1])])
	plt.plot(points, (-w[1]*points - w[0])/w[2], c="red", label="Recta RL")
	plt.legend()
	plt.title("Regresión Logística con SGD (test)")
	plt.gcf().canvas.set_window_title('Ejercicio 2 - Apartado 2b)')
	plt.show()

	# Mostramos cálculos de porcentaje de aciertos y Eout
	print("   Aciertos (test): {}%".format(get_porc(datos_test, labels_test, w)))
	print("   Eout: {}".format(Err_RL(datos_test, labels_test, w)))

	input("\n--- Pulsar tecla para continuar ---")

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n##############################")
	print("###   2 MODELOS LINEALES   ###")
	print("##############################")
	apartado1()
	apartado2()

if __name__ == "__main__":
	main()
