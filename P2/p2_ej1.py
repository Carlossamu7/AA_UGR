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
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#

#------------------------------ Apartado 1 -------------------------------------#

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
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
