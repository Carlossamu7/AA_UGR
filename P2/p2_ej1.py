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
#-------------- Ejercicio sobre la complejidad de H y el ruido -----------------#
#-------------------------------------------------------------------------------#

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
	media = 0
	out = np.zeros((N,dim),np.float64)
	for i in range(N):
		# Para cada columna dim se emplea un sigma determinado. Es decir, para la
		# primera columna se usará una N(0,sqrt(sigma[0])) y para la segunda N(0,sqrt(sigma[1]))
		out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

	return out

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

#------------------------------ Apartado 1 -------------------------------------#

""" Función que ejecuta todo el apartado 1 """
def apartado1():
	print ("\n###  Apartado 1  ###\n")
	print("a) simula_unif(N, dim, rango) con N=50, dim=2 y rango=[-50,50].")
	x_unif = simula_unif(50, 2, [-50, 50])
	plt.scatter(x_unif[:, 0], x_unif[:, 1])
	plt.title("Nube de puntos con simula_unif")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 1a)')
	plt.show()

	print("\nb) simula_gaus(N, dim, sigma) con N=50, dim=2 y sigma=[5,7].")
	x_gaus = simula_gaus(50, 2, np.array([5, 7]))
	plt.scatter(x_gaus[:, 0], x_gaus[:, 1])
	plt.title("Nube de puntos con simula_gaus")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 1b)')
	plt.show()
	input("\n--- Pulsar tecla para continuar ---")

#------------------------------ Apartado 2 -------------------------------------#

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

""" Función que ejecuta todo el apartado 2 """
def apartado2():
	print ("\n###  Apartado 2  ###\n")
	N = 50
	a, b = simula_recta([-50, 50])
	x = simula_unif(N, 2, [-50, 50])
	y = np.empty((N, ))
	for i in range(N):
		y[i] = signo(f(x[i,0], x[i,1], a, b))

	print("a) Gráfica con las etiquetas de los puntos y la recta simulada.")
	plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], label="Etiqueta -1")
	plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c="orange", label="Etiqueta 1")
	points = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
	plt.plot(points, a*points+b, c="red", label="Recta simulada")
	plt.legend()
	plt.title("Clasificando puntos con la recta simulada")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 2a)')
	plt.show()

	print("\nb) Gráfica con las etiquetas de los puntos CON RUIDO y la recta simulada.")
	y_noise = np.copy(y)	# Introducimos ruido en el 10%
	ind = np.random.choice(N, size=int(N/10), replace=False)
	for i in ind:
		y_noise[i] = -y[i]

	plt.scatter(x[y_noise == -1][:, 0], x[y_noise == -1][:, 1], label="Etiqueta -1")
	plt.scatter(x[y_noise == 1][:, 0], x[y_noise == 1][:, 1], c="orange", label="Etiqueta 1")
	plt.plot(points, a*points+b, c="red", label="Recta simulada")
	plt.legend()
	plt.title("Clasificando puntos (con ruido) con la recta simulada")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 2b)')
	plt.show()

	input("\n--- Pulsar tecla para continuar ---")
	return x, y_noise

#------------------------------ Apartado 3 -------------------------------------#

""" Función en dos variables que representa una elipse
- x: primera variable de la función.
- y: segunda variable de la función."""
def f1(x, y):
	return (x-10)**2 + (y-20)**2 - 400

""" Función en dos variables que representa una elipse
- x: primera variable de la función.
- y: segunda variable de la función."""
def f2(x, y):
	return 0.5 * (x+10)**2 + (y-20)**2 - 400

""" Función en dos variables que representa una elipse
- x: primera variable de la función.
- y: segunda variable de la función."""
def f3(x, y):
	return 0.5 * (x-10)**2 - (y+20)**2 - 400

""" Función en dos variables que representa una parábola
- x: primera variable de la función.
- y: segunda variable de la función."""
def f4(x, y):
	return y - 20*x**2 - 5*x + 3

""" Para cada función pasada por argumento visualiza los puntos (x) con sus etiquetas (y)
y la gráfica de la función como frontera de clasificación.
- x: vector de puntos 2D que son las características.
- y: vector de etiquetas.
- fun: función a representar.
- title: título de la función."""
def print_graf(x, y, fun, title=""):
	plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], label="Etiqueta -1")
	plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c="orange", label="Etiqueta 1")
	#Generamos el contorno de fun
	x1, y1 = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))
	contorno = fun(x1,y1)
	plt.contour(x1, y1, contorno, [0], colors='red')
	plt.title("Clasificando puntos (con ruido) con la " + title)
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 3')
	plt.legend()
	plt.show()

""" Calcula el porcentaje de puntos bien clasificados
- datos: datos.
- labels: etiquetas.
- fun: función clasificadora."""
def get_porc(datos, labels, fun):
	signos = labels*fun(datos[:, 0], datos[:, 1])
	return 100*len(signos[signos >= 0])/len(labels)

""" Función que ejecuta todo el apartado 3 """
def apartado3(x, y):
	print ("\n###  Apartado 3  ###\n")
	print_graf(x, y, f1, "Elipse1")
	print("Acierto para '{}': {}%".format("Elipse1", get_porc(x, y, f1)))
	print_graf(x, y, f2, "Elipse2")
	print("Acierto para '{}': {}%".format("Elipse2", get_porc(x, y, f2)))
	print_graf(x, y, f3, "Elipse3")
	print("Acierto para '{}': {}%".format("Elipse3", get_porc(x, y, f3)))
	print_graf(x, y, f4, "Parábola")
	print("Acierto para '{}': {}%".format("Parábola", get_porc(x, y, f4)))
	input("\n--- Pulsar tecla para continuar ---")

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n###########################################################")
	print("###  1 EJERCICIO SOBRE LA  COMPLEJIDAD DE H Y EL RUIDO  ###")
	print("###########################################################")
	apartado1()
	x, y = apartado2()
	apartado3(x, y)

if __name__ == "__main__":
	main()
