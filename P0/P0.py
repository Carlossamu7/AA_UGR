# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import math

#######################
###   EJERCICIO 1   ###
#######################

""" Ejecución del ejercicio 1.
    Lee la base de datos de Iris. Devuelve las características y clases. """
def ejercicio1():
    print("----- Ejercicio 1 -----")

    # Leemos la base de datos de Iris
    iris = datasets.load_iris()

    # Obtenemos las 2 últimas características y clases
    x = iris.data[:, -2:];
    y = iris.target

    # Visualizamos con Scatter plot
    plt.scatter(x[0:50:,0],x[0:50:,1], label="Setosa", c="red")
    plt.scatter(x[50:100:,0],x[50:100:,1], label="Versicolor", c="green")
    plt.scatter(x[100::,0],x[100::,1], label="Virginica", c="blue")
    plt.xlabel("Largo del pétalo")
    plt.ylabel("Ancho del pétalo")
    plt.legend(loc="lower right")
    plt.title("Base de datos de Iris")
    plt.gcf().canvas.set_window_title('Ejercicio 1')
    plt.show()

    return x, y

#######################
###   EJERCICIO 2   ###
#######################

""" Ejecución del ejercicio 2. Divide train y test al 80-20.
- x: características.
- y: clases.
"""
def ejercicio2(x, y):
    print("----- Ejercicio 2 -----")

    # Separamos train (80%) y test (20%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=7)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for index_train, index_test in sss.split(x, y):
        x_train, x_test = x[index_train], x[index_test]
        y_train, y_test = y[index_train], y[index_test]

    # Imprimo los conjuntos train y test
    print("x_train:\n", x_train)
    print("\nx_test:\n", x_test)
    print("\ny_train:\n", y_train)
    print("\ny_test:\n", y_test)


#######################
###   EJERCICIO 3   ###
#######################

""" Ejecución del ejercicio 3.
    Gráficas de sin(x), cos(x) y sin(x) + cos(x) con x en [0,2PI]"""
def ejercicio3():
    print("----- Ejercicio 3 -----")

    # Obtenemos 100 valores equiespaciados entre 0 y 2PI
    puntos = np.linspace(0, 2*math.pi, num=100)
    print("100 valores equiespaciados entre 0 y 2PI:\n", puntos)

    # Calculamos imágenes de sin(x), cos(x) y sin(x)+cos(x) con x en puntos
    sen = []; cos = []; sen_cos = []
    for i in puntos:
        sen.append(math.sin(i))
        cos.append(math.cos(i))
        sen_cos.append(math.sin(i) + math.cos(i))

    # Imprimo por pantalla los valores
    print("\nSeno de los valores:\n", sen)
    print("\nCoseno de los valores:\n", cos)
    print("\nSuma de seno y coseno de los valores:\n", sen_cos)

    # Visualizamos
    plt.plot(puntos, sen, 'k--', label="sin(x)")
    plt.plot(puntos, cos, 'b--', label="cos(x)")
    plt.plot(puntos, sen_cos, 'r--', label="sin(x) + cos(x)")
    plt.xlabel("Eje x")
    plt.ylabel("Eje y")
    plt.legend(loc="lower left")
    plt.title("Imágenes de funciones en [0,2PI]")
    plt.gcf().canvas.set_window_title('Ejercicio 3')
    plt.show()


#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    x, y = ejercicio1();
    input("Pulsa 'Enter' para continuar\n")
    ejercicio2(x, y);
    input("Pulsa 'Enter' para continuar\n")
    ejercicio3();

if __name__ == "__main__":
    main()
