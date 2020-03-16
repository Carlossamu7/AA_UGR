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
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#

#------------------------------ Apartado 1 -------------------------------------#

""" Gradiente descendente. Devuelve el mínimo el nº de iteraciones usadas.
- w: vector de pesos inicial.
- lr: tasa de aprendizaje.
- grad_fun: gradiente de 'fun'.
- fun: función (diferenciable) a minimizar.
- epsilon: máximo error permitido.
- max_iters: máximo número de iteraciones.
"""
def gd(w, lr, grad_fun, fun, epsilon=-math.inf, max_iters=100000):
	it = 0
	while fun(w)>epsilon and it<max_iters:
		w = w - lr*grad_fun(w)
		it += 1
	return w, it

""" Función que ejecuta todo el apartado 1 """
def apartado1():
	print ("\n###  Apartado 1  ###\n")
	print("def gd(w, lr, grad_fun, fun, epsilon=-math.inf, max_iters=100000):")
	print("  it = 0")
	print("  while fun(w)>epsilon and it<max_iters:")
	print("    w = w - lr*grad_fun(w)")
	print("    it += 1")
	print("  return w, it")
	input("--- Pulsar tecla para continuar ---")

#------------------------------ Apartado 2 -------------------------------------#

""" Función E del apartado 2 """
def E(w):
	return (w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0]))**2

""" Derivada parcial de E respecto de u """
def Eu(w):
	return 2 * (w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0])) * (np.exp(w[1]) + 2*w[1]*np.exp(-w[0]))

""" Derivada parcial de E respecto de v """
def Ev(w):
	return  2 * (w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0])) * (w[0]*np.exp(w[1])-2*np.exp(-w[0]))

""" Gradiente de E """
def gradE(w):
	return np.array([Eu(w), Ev(w)])

""" Función que ejecuta todo el apartado 2 """
def apartado2():
	print ("\n###  Apartado 2  ###\n")
	w, num_ite = gd(np.array([1.0, 1.0]), 0.1, gradE, E, 1e-14)
	print("a) grad E(u,v) = [2 * (u*exp(v) - 2*v*exp(-u)) * (exp(v) + 2*v*exp(-u)),")
	print("                  2 * (u*exp(v) - 2*v*exp(-u)) * (u*exp(v)-2*exp(-u))]")
	input("--- Pulsar tecla para continuar ---")
	print("\nb) Numero de iteraciones: {}".format(num_ite))
	input("--- Pulsar tecla para continuar ---")
	print("\nc) Coordenadas obtenidas: ({}, {}).".format(w[0], w[1]))
	print("   Valor en el punto: {}.".format(E(w)))
	input("--- Pulsar tecla para continuar ---")

#------------------------------ Apartado 3 -------------------------------------#

""" Función f del apartado 3 """
def f(w):
	return (w[0]-2)**2 + 2*(w[1]+2)**2 + 2*np.sin(2*np.pi*w[0])*np.sin(2*np.pi*w[1])

""" Derivada parcial de f respecto de x """
def fx(w):
	return 2*(w[0]-2) + 4*np.pi*np.cos(2*np.pi*w[0])*np.sin(2*np.pi*w[1])

""" Derivada parcial de f respecto de y """
def fy(w):
	return 4*(w[1]+2) + 4*np.pi*np.sin(2*np.pi*w[0])*np.cos(2*np.pi*w[1])

""" Gradiente de f """
def gradf(w):
	return np.array([fx(w), fy(w)])

""" a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1 """
""" Función de GD que almacena los resultados para construir una gráfica. """
def gd_grafica(w, lr, grad_fun, fun, max_iters = 100000):
	it = 0
	graf = np.zeros(max_iters)
	while it < max_iters:
		graf[it] = fun(w)	# Guardamos el resultado de la iteración
		w = w - lr*gradf(w)
		it += 1

	# Dibujamos la gráfica
	plt.plot(range(0,max_iters), graf, 'b-o', label=r"$\eta$ = {}".format(lr))
	plt.xlabel('Iteraciones'); plt.ylabel('f(x,y)')
	plt.legend()
	plt.title("Curva de gradiente descendente")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 3a)')
	plt.show()
	return graf

""" b) Obtener el valor minimo y los valores de (x,y) con los
 puntos de inicio siguientes: """
""" Usando GD muestra el punto inicial, el mínimo y el valor del mínimo """
def print_gd(w, lr, grad_fun, fun, epsilon, max_iters = 1000000000):
	print("\n   Punto de inicio: ({}, {})".format(w[0], w[1]))
	w, _ = gd(w, lr, grad_fun, fun, epsilon, max_iters)
	print("   (x,y) = ({} , {})".format(w[0], w[1]))
	print("   Valor minimo: {}".format(f(w)))

""" Función que ejecuta todo el apartado 3 """
def apartado3():
	print ("\n###  Apartado 3  ###\n")

	print ("a) Gráfica con learning rate igual a 0.01")
	g1 = gd_grafica(np.array([1.0, -1.0]), 0.01, gradf, f, 50)
	print ("   Gráfica con learning rate igual a 0.1")
	g2 = gd_grafica(np.array([1.0, -1.0]), 0.1, gradf, f, 50)
	# Comparamos las gráficas
	print ("   Comparación de las dos gráficas anteriores")
	plt.plot(g1, 'b-o', label=r"$\eta$ = {}".format(0.01))
	plt.plot(g2, 'r-o', label=r"$\eta$ = {}".format(0.1))
	plt.xlabel('Iteraciones'); plt.ylabel('f(x,y)');
	plt.legend()
	plt.title("Comparación de las curvas de gradiente descendente")
	plt.gcf().canvas.set_window_title('Ejercicio 1 - Apartado 3a)')
	plt.show()
	input("--- Pulsar tecla para continuar ---")

	print("\nb) Mínimo y valor donde se alcanza según el punto inicial:")
	lrate = 0.01; eps = 0.0001; max_it = 1000
	print_gd(np.array([2.1, -2.1]), lrate, gradf, f, eps, max_it)
	print_gd(np.array([3.0, -3.0]), lrate, gradf, f, eps, max_it)
	print_gd(np.array([1.5, 1.5]), lrate, gradf, f, eps, max_it)
	print_gd(np.array([1.0, -1.0]), lrate, gradf, f, eps, max_it)
	input("--- Pulsar tecla para continuar ---")

#------------------------------ Apartado 4 -------------------------------------#
""" Función que ejecuta todo el apartado 4 """
def apartado4():
	print ("\n###  Apartado 4  ###\n")
	print ("Explicación en la memoria. Muy brevemente:")
	print("- Diferenciabilidad de la función.")
	print("- Convexidad nos asgura encotrar el mínimo global.")
	print("- Fundamental: elección de la tasa de aprendizaje y el pto inical.")
	print("- Hacer una batería de pruebas para la elección cuando la función sea compleja.\n")

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n############################################################")
	print("###  1 EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS  ###")
	print("############################################################")
	apartado1()
	apartado2()
	apartado3()
	apartado4()

if __name__ == "__main__":
	main()
