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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
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

""" Devuelve las cabeceras del problema.
"""
def get_headers():
	return ['state', 'county', 'community', 'communityname', 'fold', 'population', 'householdsize',
          'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21',
          'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
          'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire',
          'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
          'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade',
          'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ',
          'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
          'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
          'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent',
          'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
          'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam',
          'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
          'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
          'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
          'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
          'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
          'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
          'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
          'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
          'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
          'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
          'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
          'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
          'ViolentCrimesPerPop']

""" Muestra información y estadísticas de los datos.
- X: características.
- y: etiquetas.
- statistic (op): si mostrar estadísticas de los datos. Por defecto "True".
"""
def data_info(X, y, statistic=True):
	print("INFORMACIÓN DE LOS DATOS:")
	print("Número de atributos: {} (uno es el goal)".format(len(X[0])+1))
	print("Número de datos perdidos: {}".format(len(X[X=='?'])))
	outliers = np.array([('?' in X[:, i]) for i in range(len(X[0]))])
	print("Número de atributos que contienen datos perdidos: {}".format(np.count_nonzero(outliers==True)))
	#print(outliers)
	print("Intervalo en el que están las características: [{},{}]".format(np.min(X), np.max(X)))
	print("Intervalo en el que están las etiquetas: [{},{}]".format(np.min(y), np.max(y)))

	if(statistic):
		print("Tipos y cantidad de ellos en los atributos:")
		tab = [["Nominal", "Numeric", "String", "Decimal", "Semiboolean"], [1, 3, 1, 122, 1]]
		print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))

		tab = [["Intervalo", "Núm. instancias"],["[0.0, 0.1)", 0],["[0.1, 0.2)", 0],["[0.2, 0.3)", 0],["[0.3, 0.4)", 0],
			["[0.4, 0.5)", 0],["[0.5, 0.6)", 0],["[0.6, 0.7)", 0],["[0.7, 0.8)", 0],["[0.8, 0.9)", 0],["[0.9, 1.0]", 0]]
		num = []
		for i in range(0,10):
			num.append(len(y[(i/10<=y) & (y<(i+1)/10)]))
		num[-1] += len(y[y==1])
		for i in range(1, len(tab)):
			tab[i][1] = str(num[i-1]) + "  (" + str(round(100*num[i-1]/len(y), 2)) + "%)"
		print("\nNúmero de instancias de cada dígito")
		print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))

		plt.bar([tab[i][0] for i in range(1,len(tab))], num, align="center")
		plt.xlabel("Intervalo")
		plt.ylabel("Núm. instancias")
		plt.title("Gráfica de barras de las etiquetas")
		plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
		plt.show()

#---------------------- Dividiendo en 'train' y 'test' -------------------------#

""" Da información acerca de la separación realizada en 'train' y 'test'.
- y_train: etiquetas de 'train'.
- y_test: etiquetas de 'test'.
"""
def split_info(y_train, y_test):
	size_train = len(y_train)
	size_test = len(y_test)
	train_perc = 100 * size_train / (size_train+size_test)
	test_perc = 100 * size_test / (size_train+size_test)
	print("Núm. instancias: {} (train) {} (test)".format(size_train, size_test))
	print("Porcentaje (%): {} (train) {} (test)".format(round(train_perc, 3), round(test_perc, 3)))

	tab = [["Intervalo", "Núm. instancias 'train'", "Núm. instancias 'test'"],["[0.0, 0.1)", 0, 0],["[0.1, 0.2)", 0, 0],["[0.2, 0.3)", 0, 0],
		["[0.3, 0.4)", 0, 0], ["[0.4, 0.5)", 0, 0],["[0.5, 0.6)", 0, 0],["[0.6, 0.7)", 0, 0],["[0.7, 0.8)", 0, 0],["[0.8, 0.9)", 0, 0],["[0.9, 1.0]", 0, 0]]
	num_train = []
	num_test = []
	for i in range(0,10):
		num_train.append(len(y_train[(i/10<=y_train) & (y_train<(i+1)/10)]))
		num_test.append(len(y_test[(i/10<=y_test) & (y_test<(i+1)/10)]))
	num_train[-1] += len(y_train[y_train==1])
	num_test[-1] += len(y_test[y_test==1])
	for i in range(1, len(tab)):
		tab[i][1] = str(num_train[i-1]) + "  (" + str(round(100*num_train[i-1]/len(y_train), 2)) + "%)"
		tab[i][2] = str(num_test[i-1]) + "  (" + str(round(100*num_test[i-1]/len(y_test), 2)) + "%)"
	print("\nNúmero de instancias de cada dígito para 'train' y 'test'")
	print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))

	plt.bar([tab[i][0] for i in range(1,len(tab))], num_train, align="center")
	plt.xlabel("Intervalo")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de las etiquetas")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

	plt.bar([tab[i][0] for i in range(1,len(tab))], num_test, align="center")
	plt.xlabel("Intervalo")
	plt.ylabel("Núm. instancias")
	plt.title("Gráfica de barras de las etiquetas")
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	plt.show()

#------------------------------- Preprocesado ----------------------------------#

""" Preprocesamiento de los missing values. Devuelve el nuevo X y sus cabeceras.
- X: características.
- y: etiquetas.
- headers: cabeceras.
"""
def preprocess_missing_values(X, y, headers):
	print("Índices de atributos no predictivos:")
	to_delete = [0,1,2,3,4]
	print(to_delete)
	print("Cabeceras de los atributos eliminados por ser no predictivos:")
	print([headers[h] for h in to_delete])
	print("Eliminando atributos no predictivos...")
	newheaders = np.delete(headers, to_delete)
	newX = np.delete(X, to_delete, axis=1)
	newX[newX=='?'] = np.nan
	newX = newX.astype('float64')
	input("--- Pulsar tecla para continuar ---\n")

	np.set_printoptions(suppress=True)
	outliers = np.array([np.count_nonzero(np.isnan(newX[:, i])) for i in range(len(newX[0]))], dtype=int)
	perc = np.empty(outliers.shape, dtype='float')
	for i in range(len(perc)):
		perc[i] = round(100 * outliers[i] / len(newX), 2)
	perc_extracted_pos = np.where(perc>0)[0]
	num_extracted = [outliers[i] for i in perc_extracted_pos]
	perc_extracted = [perc[i] for i in perc_extracted_pos]
	headers_extracted = [newheaders[i] for i in perc_extracted_pos]
	tab = [["Cabecera", "Val. perdidos", "Porcentaje (%)"]]
	for i in range(len(num_extracted)):
		tab.append([headers_extracted[i], num_extracted[i], perc_extracted[i]])
	print("Mostrando atributos que tienen valores perdidos:")
	print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))
	#print("Número de datos perdidos por atributo:")
	#print(outliers)
	#print("\nPorcentaje de datos perdidos por atributo:")
	#print(perc)
	input("--- Pulsar tecla para continuar ---\n")

	index = np.where(perc>10)[0]
	print("Índices de atributos con más de un 10% de datos perdidos:")
	print(index+5)
	print("\nCabeceras de los atributos eliminados por ser datos perdidos:")
	print([newheaders[h] for h in index])
	print("Eliminando atributos no predictivos...")
	newheaders = np.delete(newheaders, index)
	newX = np.delete(newX, index, axis=1)
	input("--- Pulsar tecla para continuar ---\n")

	to_treat_index = np.where((perc>0) & (perc<=10))[0]
	print("\nÍndices de atributos con menos de un 10% de datos perdidos que hay que tratar:")
	print(to_treat_index)
	print("Haciendo la media de las instancias del mismo estado para dicho atributo...")
	for i in to_treat_index:
		col = newX[:, i]
		pos = np.where(np.isnan(col))[0]
		print("   Datos perdidos en las posiciones: {}".format(pos))
		for p in pos:
			state = X[p][0]
			print("   'State' del 'outlier': " + state)
			sum = 0
			count = 0
			for j in range(len(X)):
				if(X[j][0]==state and j!=p):
					sum += newX[j][i]
					count += 1
			newX[p][i] = round(sum/count,4)
			print("   Hay {} coincidencias de estado y se inserta el valor medio: {}".format(count, newX[p][i]))

	print("\nEl número de atributos ha pasado de {} a {}".format(len(headers), len(newheaders)))
	return newX, newheaders

""" Muestra matriz de correlación de los datos.
- data: datos.
- title (op): título. Por defecto "".
"""
def show_preprocess(data, title=""):
	fig, axs = plt.subplots()
	corr_matrix = np.abs(np.corrcoef(data.T))
	im = axs.matshow(corr_matrix, cmap="GnBu")
	fig.suptitle(title)
	plt.gcf().canvas.set_window_title("Práctica 3 - Clasificación")
	fig.colorbar(im, ax=axs, shrink=0.6)
	plt.show()

#------------------------------ Clasificadores ---------------------------------#

""" Función para crear una lista de modelos basados en SGDR para diferentes valores
de alpha y tol sobre los datos preprocesados. Devuelve dicha lista.
- alphas: lista de valores de alpha.
- tols: lista de tolerancias.
"""
def SGD_regressors(alphas, tols):
	models = []
	for al in alphas:
		for to in tols:
			models.append(SGDRegressor(loss="squared_loss", penalty="l2", alpha=al, tol=to))
	return models

""" Funcion para evaluar una lista de modelos.
Devuelve los scores obtenidos.
- models: modelos en formato lista.
- X_train: características del conjunto de entrenamiento.
- y_train: etiquetas del conjunto de entrenamiento.
- X_test: características del conjunto de test.
- y_test: etiquetas del conjunto de test.
"""
def models_eval(models, X_train, y_train, X_test, y_test):
	dats = []		# medias de las cv de cada modelo.

	# Se evalúan los modelos actualizando dichas listas.
	for model in models:
		model.fit(X_train, y_train)
		dats.append([model.score(X_train, y_train), model.score(X_test, y_test)])

	return dats

""" Imprime la tabla de los scores. Devuelve dicha tabla.
- alphas: lista de valores de alpha.
- tols: lista de tolerancias.
- scores: medida R^2 en 'train' y 'test'.
"""
def info_scores(alphas, tols, scores):
	tab = [["Modelo", "Alpha", "Tol.", "R^2 'train'", "R^2 'test'"]]

	for i in range(len(alphas)):
		for j in range(len(tols)):
			tab.append(["SGDR", alphas[i], tols[j], scores[i*len(tols)+j][0], scores[i*len(tols)+j][1]])

	print("Mostrando tabla con los 'scores':")
	print(tabulate(tab, headers='firstrow', numalign='center', stralign='center', tablefmt='fancy_grid'))
	return tab

""" Elije el mejor modelo de la tabla.
- tab: tabla.
"""
def select_best(tab):
	col = [tab[i][4] for i in range(1,len(tab))]
	maxim = max(col)
	pos = np.where(col==maxim)[0] + 1
	print("El mejor R^2 es: {}".format(maxim))
	print("El MEJOR MODELO es: {} (alpha={} y tol={})".format(tab[pos[0]][0], tab[pos[0]][1], tab[pos[0]][2]))
	return pos[0]

#---------------------------------- Errores ------------------------------------#

""" Muestra información y estadísticas de los datos.
- model: modelo.
- scores: medida R^2 en 'train' y 'test'.
- mse: error cuadrático medio.
- title (op): título del modelo. Por defecto "".
"""
def show_errors(model, scores, mse, title=""):
	print(model)
	print("R^2 del modelo en 'train': {:.5f}".format(scores[0]))	# R^2
	print("R^2 del modelo en 'test': {:.5f}".format(scores[1]))		# R^2
	print("Error cuadrático medio: {}".format(mse))


########################
#####     MAIN     #####
########################

""" Programa principal. """
def main():
	print("\n###############################")
	print("#########  REGRESIÓN  #########")
	print("###############################")

	print("\nLeyendo datos de 'communities'.\n")
	X, y = read_data("datos/communities.data", ",")
	headers = get_headers()
	y = y.astype('float64')
	data_info(X, y)
	input("--- Pulsar tecla para continuar ---\n")

	print("PREPROCESADO DE DATOS PERDIDOS\n")
	newX, headers = preprocess_missing_values(X, y, headers)
	input("--- Pulsar tecla para continuar ---\n")
	data_info(newX, y, False)
	input("--- Pulsar tecla para continuar ---\n")

	print("Separando en 'train' y 'test' los datos de 'communities'")
	X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.25, random_state=1)
	split_info(y_train, y_test)
	input("--- Pulsar tecla para continuar ---\n")

	print("PREPROCESANDO LOS DATOS")
	preprocess = [("var", VarianceThreshold(threshold=0.0))]
	preprocessor = Pipeline(preprocess)
	X_train_preprocess = preprocessor.fit_transform(X_train)
	print("Número de características con varianza cero: {}".format(X_train.shape[1] - X_train_preprocess.shape[1]))
	input("--- Pulsar tecla para continuar ---\n")

	print("Imprimiendo matriz de correlación después de preprocesar los datos.")
	show_preprocess(X_train_preprocess, "Regresión de 'communities' - Matriz de correlación")
	input("--- Pulsar tecla para continuar ---\n")

	# Obtener valores medios y desviaciones de las evaluaciones
	print("EVALUANDO DIFERENTES MODELOS")
	print("- Seis modelos 'SGDRegressor'.")
	alphas = [0.00001, 0.0001, 0.001]
	tols = [1e-9, 1e-8, 1e-7]
	# Creación de los modelos
	models = SGD_regressors(alphas, tols)
	# Evalúo los modelos
	scores = models_eval(models, X_train, y_train, X_test, y_test)
	tab = info_scores(alphas, tols, scores)
	input("--- Pulsar tecla para continuar ---\n")

	print("ELECCIÓN DEL MEJOR MODELO")
	pos = select_best(tab)-1
	# Mostrando el mejor modelo, su matriz de confusión y sus errores.
	models[pos].fit(X_train, y_train)
	y_pred = models[pos].predict(X_test)
	mse = mean_squared_error(y_pred, y_test)
	show_errors(models[pos], scores[pos], mse, "SGDR (alpha={}, tol={})".format(tab[pos+1][1], tab[pos+1][2]))
	input("--- Pulsar tecla para continuar ---\n")

if __name__ == "__main__":
	main()
