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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import warnings		# Quita warnings
warnings.filterwarnings('ignore')

# Fijamos la semilla
np.random.seed(1)

#-------------------------------------------------------------------------------#
#---------------------------------- Regresión ----------------------------------#
#-------------------------------------------------------------------------------#

#---------------------------------- Lectura ------------------------------------#

""" Carga datos leyendo de un fichero de texto.
- filename: fichero a leer.
- separator (op): El elemento que separa los datos.
"""
def read_data(filename, separator):
	data = np.loadtxt(filename, delimiter=separator, dtype=object)
	return data[:, :-1], data[:, -1]

""" Muestra información y estadísticas de los datos.
- X: características.
- y: etiquetas.
"""
def data_info(X, y):
	print("\nINFORMACIÓN DE LOS DATOS:")
	print("Número de atributos: {}".format(len(X[0])+1))
	print("Núm. de outliers: {}".format(len(X[X=='?'])))
	outliers = np.array([('?' in X[:, i]) for i in range(len(X[0]))])
	print("Núm. atributos que contienen outliers: {}".format(np.count_nonzero(outliers==True)))
	print(outliers)
	print("Tipos y cantidad de ellos en los atributos:")
	tab = [["Nominal", "Numeric", "String", "Decimal", "Semiboolean"], [1, 3, 1, 122, 1]]
	print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))
	print("Intervalo en el que están las características: [{},{}]".format(np.min(X), np.max(X)))
	print("Intervalo en el que están las etiquetas: [{},{}]".format(np.min(y), np.max(y)))
	tab = [["Dígito", "Instancias 'train'", "Instancias 'test'"]]
	num_train = [[], []]
	for i in range(np.min(y_train), np.max(y_train)+1):
		num_train[0].append(len(y_train[y_train==i]))
		num_train[1].append(round(100*len(y_train[y_train==i])/len(y_train), 2))
	print("\nNúmero de instancias de cada dígito para 'train' y 'test'")
	print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))

	plt.bar([0,1,2,3,4,5,6,7,8,9], num_train[0], align="center")
	plt.xlabel("Dígitos")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de los datos de 'train'")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

#---------------------- Dividiendo en 'train' y 'test' -------------------------#

def split_info(X_train, X_test):
	size_train = X_train.shape[0]
	size_test = X_test.shape[0]
	train_perc = 100 * size_train / (size_train+size_test)
	test_perc = 100 * size_test / (size_train+size_test)
	print("Núm. instancias: {} (train) {} (test)".format(size_train, size_test))
	print("Porcentaje (%): {} (train) {} (test)".format(round(train_perc, 3), round(test_perc, 3)))

#------------------------------- Preprocesado ----------------------------------#

""" Muestra matriz de correlación de los datos antes y después del preprocesado.
- data: datos originales.
- preprocess_data: datos preprocesados.
- title (op): título. Por defecto "".
"""
def show_preprocess(data, preprocess_data, title=""):
	fig, axs = plt.subplots(1, 2, figsize=[12.0, 5.8])

	corr_matrix = np.abs(np.corrcoef(data.T))
	im = axs[0].matshow(corr_matrix, cmap="GnBu")
	axs[0].title.set_text("Antes del preprocesado")

	corr_matrix_post = np.abs(np.corrcoef(preprocess_data.T))
	axs[1].matshow(corr_matrix_post, cmap="GnBu")
	axs[1].title.set_text("Después del preprocesado")

	fig.suptitle(title)
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
	plt.show()

#------------------------------ Clasificadores ---------------------------------#

""" Función para crear una lista de pipelines con el modelo RL para diferentes valores de C
sobre los datos preprocesados. Devuelve dicha lista.
- Cs: Lista de valores C.
"""
def RL_clasificators(Cs):
    # Inicializando lista de Pipeline
    pipes = []

    for c in Cs:	# Para cada C se inserta un modelo.
        pipes.append(Pipeline([("var", VarianceThreshold(threshold=0.0)),
								   ("scaled", StandardScaler()),
								   ("PCA", PCA(n_components=0.95)),
								   ("svm",  LogisticRegression(C=c, multi_class='multinomial'))]))
    return pipes

""" Función para crear una lista de pipelines con el modelo SVM para diferentes valores de C
sobre los datos preprocesados. Devuelve dicha lista.
- Cs: Lista de valores C.
"""
def SVM_clasificators(Cs):
    # Inicializando lista de Pipeline
    pipes = []

    for c in Cs:	# Para cada C se inserta un modelo.
        pipes.append(Pipeline([("var", VarianceThreshold(threshold=0.0)),
								   ("scaled", StandardScaler()),
								   ("PCA", PCA(n_components=0.95)),
								   ("log",  LinearSVC(C=c, random_state=1, loss='hinge', multi_class="crammer_singer"))]))
    return pipes

""" Funcion para evaluar una lista de modelos.
Devuelve las medias y desv típicas de cada modelo.
- models: modelos en formato lista.
- X: características.
- y: etiquetas.
- cv (op): parámetro de validación cruzada. Por defecto 5.
"""
def models_eval(models, X, y, cv=5):
    means = []		# medias de las cv de cada modelo.
    devs = []		# desv. típicas de las cv de cada modelo.

    # Se evalúan los modelos actualizando dichas listas.
    for model in models:
        results = cross_val_score(model, X, y, scoring="accuracy", cv=cv)
        means.append(abs(results.mean()))		# Valor absoluto de la media
        devs.append(np.std(results))			# Guardar desviaciones

    return means, devs

#---------------------- Matrices de confución y errores ------------------------#

""" Muestra matriz de confusión.
- y_real: etiquetas reales.
- y_pred: etiquetas predichas.
- norm (op): indica si normalizar (dar en %) la matriz de confusión. Por defecto 'True'.
"""
def show_confussion_matrix(y_real, y_pred, mtype, norm=True):
	mat = confusion_matrix(y_real, y_pred)
	if(norm):
		mat = 100*mat.astype("float64")/mat.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	ax.matshow(mat, cmap="GnBu")
	ax.set(title="Matriz de confusión para {}".format(mtype),
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
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

""" Muestra información y estadísticas de los datos.
- model: modelo.
- X_train: características del conjunto de entrenamiento.
- y_train: etiquetas del conjunto de entrenamiento.
- X_test: características del conjunto de test.
- y_test: etiquetas del conjunto de test.
- title (op): título del modelo. Por defecto "".
"""
def show_confussion_errors(model, X_train, y_train, X_test, y_test, title=""):
	print("MEJOR MODELO: '" + title + "'. De ahora en adelante se usa éste.")
	print(model)
	print("Entrenando el modelo con el conjunto 'train'.")
	model.fit(X_train, y_train)
	print("Haciendo las predicciones sobre 'test'")
	y_pred = model.predict(X_test)
	print("Mostrando matriz de confusión sin normalizar.")
	show_confussion_matrix(y_test, y_pred, title, False)
	print("Mostrando matriz de confusión normalizada.")
	show_confussion_matrix(y_test, y_pred, title)
	print("Error del modelo en 'train': {:.5f}".format(1 - model.score(X_train, y_train)))
	print("Error del modelo en 'test': {:.5f}".format(1 - model.score(X_test, y_test)))


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n###############################")
	print("#########  REGRESIÓN  #########")
	print("###############################")

	print("Leyendo datos de 'communities'.")
	X, y = read_data("datos/communities.data", ",")
	data_info(X, y)
	input("--- Pulsar tecla para continuar ---\n")

	print("Separando en 'train' y 'test' los datos de 'communities'")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
	split_info(X_train, X_test)
	input("--- Pulsar tecla para continuar ---\n")

if __name__ == "__main__":
	main()
