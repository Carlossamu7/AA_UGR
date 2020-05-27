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
#---------------------------------- Regresión ----------------------------------#
#-------------------------------------------------------------------------------#

#---------------------------------- Lectura ------------------------------------#

""" Carga datos leyendo de un fichero de texto.
- filename: fichero a leer.
- separator (op): El elemento que separa los datos.
"""
def read_split_data(filename, separator):
	data = np.loadtxt(filename, delimiter=separator, dtype=int)
	return data[:, :-1], data[:, -1]

""" Muestra información y estadísticas de los datos.
- X_train: características del conjunto de entrenamiento.
- y_train: etiquetas del conjunto de entrenamiento.
- X_test: características del conjunto de test.
- y_test: etiquetas del conjunto de test.
"""
def data_info(X_train, y_train, X_test, y_test):
	print("\nINFORMACIÓN DE LOS DATOS:")
	size_train = X_train.shape[0]
	size_test = X_test.shape[0]
	train_perc = 100 * size_train / (size_train+size_test)
	test_perc = 100 * size_test / (size_train+size_test)
	print("Núm. instancias: {} (train) {} (test)".format(size_train, size_test))
	print("Porcentaje (%): {} (train) {} (test)".format(round(train_perc, 3), round(test_perc, 3)))
	print("Outliers en 'train': {}".format(X_train[X_train==np.nan].sum() + y_train[y_train==np.nan].sum()))
	print("Outliers en 'test': {}".format(X_test[X_test==np.nan].sum() + y_test[y_test==np.nan].sum()))
	print("Todos los valores son enteros en 'train': {}".format(X_train.dtype==np.int64 and y_train.dtype==np.int64))
	print("Todos los valores son enteros en 'test': {}".format(X_test.dtype==np.int64 and y_test.dtype==np.int64))
	print("Intervalo en el que están las características de 'train': [{},{}]".format(np.min(X_train), np.max(X_train)))
	print("Intervalo en el que están las etiquetas de 'train': [{},{}]".format(np.min(y_train), np.max(y_train)))
	print("Intervalo en el que están las características de 'test': [{},{}]".format(np.min(X_test), np.max(X_test)))
	print("Intervalo en el que están las etiquetas de 'test': [{},{}]".format(np.min(y_test), np.max(y_test)))
	tab = [["Dígito", "Instancias 'train'", "Instancias 'test'"]]
	num_train = [[], []]
	for i in range(np.min(y_train), np.max(y_train)+1):
		num_train[0].append(len(y_train[y_train==i]))
		num_train[1].append(round(100*len(y_train[y_train==i])/len(y_train), 2))
	num_test = [[], []]
	for i in range(np.min(y_test), np.max(y_test)+1):
		num_test[0].append(len(y_test[y_test==i]))
		num_test[1].append(round(100*len(y_test[y_test==i])/len(y_test), 2))
	for i in range(np.min(y_test), np.max(y_test)+1):
		tab.append([i, str(num_train[0][i]) + "  (" + str(num_train[1][i]) + "%)", str(num_test[0][i]) + "  (" + str(num_test[1][i]) + "%)"])
	print("\nNúmero de instancias de cada dígito para 'train' y 'test'")
	print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))

	plt.bar([0,1,2,3,4,5,6,7,8,9], num_train[0], align="center")
	plt.xlabel("Dígitos")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de los datos de 'train'")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

	plt.bar([0,1,2,3,4,5,6,7,8,9], num_test[0], align="center")
	plt.xlabel("Dígitos")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de los datos de 'test'")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

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

	print("Leyendo datos y separando en 'train' y 'test' de 'optdigits'.")
	X_train, y_train = read_split_data("datos/optdigits.tra", ",")
	X_test, y_test = read_split_data("datos/optdigits.tes", ",")
	data_info(X_train, y_train, X_test, y_test)
	input("--- Pulsar tecla para continuar ---\n")

if __name__ == "__main__":
	main()
