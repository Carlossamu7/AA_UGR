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

""" Carga datos leyendo de un fichero de texto.
- filename: fichero a leer.
- separator (op): El elemento que separa los datos.
"""
def read_split_data(filename, separator):
	data = np.loadtxt(filename, delimiter=separator)
	return data[:, :-1], data[:, -1]

#---------------------------- Apartado 1 y 2 -----------------------------------#

def visualize_data(x, y, title=None):
	"""Representa conjunto de puntos 2D clasificados.
	Argumentos posicionales:
	- x: Coordenadas 2D de los puntos
	- y: Etiquetas"""

	_, ax = plt.subplots()

	# Establece límites
	xmin, xmax = np.min(x[:, 0]), np.max(x[:, 0])
	ax.set_xlim(xmin - 1, xmax + 1)
	ax.set_ylim(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1)

	# Pinta puntos
	ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", alpha=0.8)

	# Pinta etiquetas
	labels = np.unique(y)
	for label in labels:
		centroid = np.mean(x[y == label], axis=0)
		ax.annotate(int(label), centroid, size=14, weight="bold", color="white", backgroundcolor="black")

	# Muestra título
	if title is not None:
		plt.title(title)
	plt.show()


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n#################################")
	print("########  CLASIFICACIÓN  ########")
	print("#################################")
	X_train, y_train = read_split_data("datos/optdigits.tra", ",")
  	X_test, y_test = read_split_data("datos/optdigits.tes", ",")

if __name__ == "__main__":
	main()
