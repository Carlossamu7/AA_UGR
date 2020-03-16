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
#------------------------------------ Bonus ------------------------------------#
#-------------------------------------------------------------------------------#

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

""" Hessiana de f """
def hessf(x, y):
	return np.array([
		2 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
		8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
		8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
		4 - 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
	]).reshape((2, 2))

# Rescato esta función de p1_ej1.py
""" Función de GD que almacena los resultados para construir una gráfica. """
def gd_grafica(w, lr, grad_fun, fun, max_iters = 100000):
	it = 0
	graf = np.zeros(max_iters)
	while it < max_iters:
		graf[it] = fun(w)	# Guardamos el resultado de la iteración
		w = w - lr*gradf(w)
		it += 1
	return graf

""" Newton. Devuelve el mínimo el nº de iteraciones usadas.
- w: vector de pesos inicial.
- lr: tasa de aprendizaje.
- grad_fun: gradiente de 'fun'.
- fun: función (diferenciable) a minimizar.
- hess_fun: función hessiana.
- max_iters: máximo número de iteraciones.
"""
def newton(w, lr, grad_fun, fun, hess_fun, max_iters = 100000):
	w_list = [w]
	it = 0

	while it < max_iters:
		w = w - lr*np.linalg.inv(hess_fun(w[0],w[1])).dot(grad_fun(w))
		w_list.append(w)
		it += 1

	return np.array(w_list)

""" Usando GD muestra el punto inicial, el mínimo y el valor del mínimo.
- w: vector de pesos inicial.
- lr: tasa de aprendizaje.
- grad_fun: gradiente de 'fun'.
- fun: función (diferenciable) a minimizar.
- hess_fun: función hessiana.
- max_iters: máximo número de iteraciones.
"""
def print_newton(w, lr, grad_fun, fun, hess_fun, max_iters = 100000):
	print("\n   Punto de inicio: ({}, {})".format(w[0], w[1]))
	w = newton(w, lr, grad_fun, fun, hess_fun, max_iters)[-1]	# último elemento
	print("   (x,y) = ({} , {})".format(w[0], w[1]))
	print("   Valor minimo: {}".format(f(w)))

""" Función que ejecuta todo el apartado 1 """
def apartado1():
	# Representación de curva de decrecimiento
	print("\nComparación de las curvas de GD y Newton")
	g1 = gd_grafica(np.array([1.0, -1.0]), 0.01, gradf, f, 50)
	g2 = gd_grafica(np.array([1.0, -1.0]), 0.1, gradf, f, 50)
	g3 = np.apply_along_axis(f, 1, newton(np.array([1.0, -1.0]), 0.01, gradf, f, hessf, 50))
	g4 = np.apply_along_axis(f, 1, newton(np.array([1.0, -1.0]), 0.1, gradf, f, hessf, 50))
	plt.plot(g1, 'b-o', label=r"GD, $\eta$ = 0.01")
	plt.plot(g2, 'k-o', label=r"GD, $\eta$ = 0.1")
	plt.plot(g3, 'g-o', label=r"Newton, $\eta$ = 0.01")
	plt.plot(g4, 'c-o', label=r"Newton, $\eta$ = 0.1")
	plt.legend()
	plt.title("Comparación de las curvas de GD y Newton")
	plt.gcf().canvas.set_window_title('Bonus')
	plt.show()
	input("--- Pulsar tecla para continuar ---")

	# Cálculo de puntos del método de Newton
	print("\nMínimo y valor donde se alcanza según el punto inicial:")
	lrate = 0.01; eps = 0.0001; max_it = 1000
	print_newton(np.array([2.1, -2.1]), lrate, gradf, f, hessf, max_it)
	print_newton(np.array([3.0, -3.0]), lrate, gradf, f, hessf, max_it)
	print_newton(np.array([1.5, 1.5]), lrate, gradf, f, hessf, max_it)
	print_newton(np.array([1.0, -1.0]), lrate, gradf, f, hessf, max_it)
	input("--- Pulsar tecla para continuar ---\n")

########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n#############################")
	print("##########  BONUS  ##########")
	print("#############################")

	apartado1()

if __name__ == "__main__":
	main()
