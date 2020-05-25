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
from tabulate import tabulate

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV


# Fijamos la semilla
np.random.seed(1)

#-------------------------------------------------------------------------------#
#-------------------------------- Clasificación --------------------------------#
#-------------------------------------------------------------------------------#

#---------------------------------- Lectura ------------------------------------#

""" Carga datos leyendo de un fichero de texto.
- filename: fichero a leer.
- separator (op): El elemento que separa los datos.
"""
def read_split_data(filename, separator):
	data = np.loadtxt(filename, delimiter=separator, dtype=int)
	return data[:, :-1], data[:, -1]

def data_info(X_train, y_train, X_test, y_test):
	print("\INFORMACIÓN DE LOS DATOS")
	size_train = X_train.shape[0]
	size_test = X_test.shape[0]
	train_perc = 100 * size_train / (size_train+size_test)
	test_perc = 100 * size_test / (size_train+size_test)
	print("Núm. instancias: {} (train) {} (test).".format(size_train, size_test))
	print("Porcentaje (%): {} (train) {} (test)".format(round(train_perc, 3), round(test_perc, 3)))
	print("Intervalo en el que están las características de train: [{},{}]".format(np.min(X_train), np.max(X_train)))
	print("Intervalo en el que están las etiquetas de train: [{},{}]".format(np.min(y_train), np.max(y_train)))
	print("Intervalo en el que están las características de test: [{},{}]".format(np.min(X_test), np.max(X_test)))
	print("Intervalo en el que están las etiquetas de test: [{},{}]".format(np.min(y_test), np.max(y_test)))
	tab = [['Díg. 0','Díg. 1','Díg. 2','Díg. 3','Díg. 4','Díg. 5','Díg. 6','Díg. 7','Díg. 8','Díg. 9']]

	num = []
	for i in range(np.min(y_train), np.max(y_train)+1):
		num.append(len(y_train[y_train==i]))
	tab.append(num)
	print("\nNúmero de instancias de cada dígito para 'train'")
	print(tabulate(tab, headers='firstrow', tablefmt='fancy_grid'))
	plt.bar([0,1,2,3,4,5,6,7,8,9], tab[1], align="center")
	plt.xlabel("Dígitos")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de los datos de 'train'")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

	num = []
	for i in range(np.min(y_test), np.max(y_test)+1):
		num.append(len(y_test[y_test==i]))
	tab[1] = num
	print("\nNúmero de instancias de cada dígito para 'test'")
	print(tabulate(tab, headers='firstrow', tablefmt='fancy_grid'))
	plt.bar([0,1,2,3,4,5,6,7,8,9], tab[1], align="center")
	plt.xlabel("Dígitos")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de los datos de 'test'")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()


#------------------------------- Preprocesado ----------------------------------#

"""Muestra matriz de correlación para datos antes y después del preprocesado."""
def show_preprocess(data, preprocess_data, title=""):
	fig, axs = plt.subplots(1, 2, figsize=[12.0, 5.8])

	corr_matrix = np.abs(np.corrcoef(data.T))
	im = axs[0].matshow(corr_matrix, cmap="GnBu")
	axs[0].title.set_text("Antes del preprocesado")

	corr_matrix_post = np.abs(np.corrcoef(preprocess_data.T))
	axs[1].matshow(corr_matrix_post, cmap="GnBu")
	axs[1].title.set_text("Después del preprocesado")

	fig.suptitle(title)
	fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
	plt.show()


"""Muestra matriz de confusión.
Versión simplificada del ejemplo
scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
def show_confussion_matrix(y_real, y_pred, mtype, norm=True):
	mat = confusion_matrix(y_real, y_pred)
	if(norm):
		mat = 100*mat.astype("float64")/mat.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	ax.matshow(mat, cmap="GnBu")
	ax.set(title="Matriz de confusión para predictor {}".format(mtype),
		   xticks=np.arange(10), yticks=np.arange(10),
		   xlabel="Etiqueta", ylabel="Predicción")

	for i in range(10):
		for j in range(10):
			if(norm):
				ax.text(j, i, "{:.0f}%".format(mat[i, j]), ha="center", va="center",
					color="black" if mat[i, j] < 50 else "white")
			else:
				ax.text(j, i, "{:.0f}".format(mat[i, j]), ha="center", va="center",
					color="black" if mat[i, j] < 50 else "white")

	plt.show()


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n#################################")
	print("########  CLASIFICACIÓN  ########")
	print("#################################\n")

	print("Leyendo datos y separando en 'train' y 'test' de 'optdigits'.")
	X_train, y_train = read_split_data("datos/optdigits.tra", ",")
	X_test, y_test = read_split_data("datos/optdigits.tes", ",")
	data_info(X_train, y_train, X_test, y_test)
	input("--- Pulsar tecla para continuar ---\n")

	print("Preprocesando los datos.")
	preprocess = [("var", VarianceThreshold(threshold=0.0)), ("scaled", StandardScaler()), ("PCA", PCA(n_components=0.95))]
	preprocessor = Pipeline(preprocess)
	X_train_preprocess = preprocessor.fit_transform(X_train)
	print("Número de características de 'train' antes del preprocesado: {}".format(X_train.shape[1]))
	print("Número de características de 'train' después del preprocesado: {}".format(X_train_preprocess.shape[1]))
	input("--- Pulsar tecla para continuar ---\n")

	print("Imprimiendo matriz de correlación antes y después de preprocesar los datos.")
	show_preprocess(VarianceThreshold(threshold=0.0).fit_transform(X_train),
					X_train_preprocess, "Clasificación de 'optdigits'")
	input("--- Pulsar tecla para continuar ---\n")

	print("Elección de un modelo logístico.")
	classification = [("log", LogisticRegressionCV(Cs=4, penalty='l2', cv=5, scoring='accuracy',
									fit_intercept=True, multi_class='multinomial', max_iter = 2000))]
	model = Pipeline(preprocess + classification)
	print(model)
	print("Entrenando el modelo con el conjunto de 'train'.")
	model.fit(X_train, y_train)
	print("Haciendo las predicciones sobre 'test'")
	y_pred = model.predict(X_test)
	print("Mostrando matriz de confusión sin normalizar.")
	show_confussion_matrix(y_test, y_pred, "Logístico", False)
	print("Mostrando matriz de confusión normalizada.")
	show_confussion_matrix(y_test, y_pred, "Logístico")
	print("Error de del modelo logístico en training: {:.3f}".format(1 - model.score(X_train, y_train)))
	print("Error de del modelo logístico en test: {:.3f}".format(1 - model.score(X_test, y_test)))
	input("--- Pulsar tecla para continuar ---\n")

if __name__ == "__main__":
	main()
