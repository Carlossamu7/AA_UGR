# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################



#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#

# Fijamos la semilla

def E(w): 
	return """ Funcion """
			 
# Derivada parcial de E respecto de u
def Eu(w):
	return """ Derivada parcial """

# Derivada parcial de E respecto de v
def Ev(w):
	return """ Derivada parcial """
	
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

def gd(w, lr, grad_fun, fun, epsilon, max_iters = ):		
	return w, it

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

def f(w):   
	return """ Funcion """
	
# Derivada parcial de f respecto de x
def fx(w):
	return """ Derivada parcial """

# Derivada parcial de f respecto de y
def fy(w):
	return """ Derivada parcial """
	
# Gradiente de f
def gradf(w):
	return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
def gd_grafica(w, lr, grad_fun, fun, max_iters = ):
				
	plt.plot(range(0,max_iters), graf, 'bo')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.show()	

print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
print ('\nGrafica con learning rate igual a 0.1')

input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def gd(w, lr, grad_fun, fun, max_iters = ):		
	return w

print ('Punto de inicio: (2.1, -2.1)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\nprint ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")