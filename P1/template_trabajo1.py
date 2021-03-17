# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Ángel Cabeza Martín
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#Derivada parcial de E con respecto a u
def E(u,v):
    if not (apartado3):
        return (u**3*np.e**(v-2)-2*v**2*np.e**-u)**2
    elif (apartado3):
        return (u + 2)**2 + 2*(v - 2)**2 + (2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v))

#Derivada parcial de E con respecto a u (o x)
def dEu(u,v):
    if not (apartado3):
        return 2*(u**3*np.e**(v-2)-2*v**2*np.e**-u)*(3*u**2*np.e**(v-2)+2*v**2*np.e**-u)
    elif (apartado3):
        return 2*(u + 2) + 4*np.pi*np.cos(2*np.pi*u)*np.sin(2*np.pi*v)
    
#Derivada parcial de E con respecto a v (o y)
def dEv(u,v):
    if not (apartado3):
        return 2*(u**3*np.e**(v-2)-2*v**2*np.e**-u)*(u**3*np.e**(v-2)-4*v*np.e**-u)
    elif (apartado3):
        return 4*(v - 2) + 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v)
    
#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


def gradient_descent(initial_point,learning_rate,error2get,tope):
    #
    # gradiente descendente
    #
    iterations = 0
    
    w = initial_point

    while ( ( (E(w[0],w[1])) > error2get ) and (iterations < tope) ):
        w = w - learning_rate * gradE(w[0],w[1])
        
        if (apartado3):
            valor = E(w[0],w[1])
            puntos_grafica.append(E(w[0],w[1]))
            iteraciones.append(iterations)
            
        iterations = iterations + 1
    return w, iterations    


eta = 0.01
learning_rate = 0.1
maxIter = 10000000000                  # Tope muy grande porque queremos que se pare cuando llegue a un error específico
error2get = 1e-14
initial_point = np.array([1.0,1.0])
apartado3 = False
w, it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print ( '¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor deE(u,v)inferior a 10−14? ', it)
print (' \n¿En qué coordenadas(u,v) se alcanzó por primera vez un valor igual o menor a 10−14\n (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print( 'Ahora vamos a trabajar con la función f(x,y) = (x+ 2)2+ 2(y−2)2+ 2sin(2πx)sin(2πy)\n')

apartado3 = True
learning_rate = 0.01
maxIter = 50
error2get = -999999
initial_point = np.array([-1.0,1.0])
puntos_grafica = []
iteraciones = []
w, it = gradient_descent(initial_point,learning_rate,error2get,maxIter)
print ( '\nEncontrado el mínimo en las coordenadas: (', w[0], ', ', w[1],')')


puntos_funcion = np.array(puntos_grafica)
iterac = np.array(iteraciones)

plt.plot(iterac,puntos_funcion)
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función')
plt.title('Gráfica que relaciona iteraciones y valor de la función')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Vamos a repetir el experimento pero con una tasa de aprendizaje de 0.1')
learning_rate = 0.1

puntos_grafica = []
iteraciones = []

w , it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

puntos_funcion = np.array(puntos_grafica)
iterac = np.array(iteraciones)

plt.plot(iterac,puntos_funcion)
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función')
plt.title('Gráfica que relaciona iteraciones y valor de la función')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print ('Vamos a aplicar el algoritmo del gradiente con distintos puntos iniciales\n')

initial_point = np.array([-0.5,0.5])
learning_rate = 0.01

w,it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print("Con [-0.5,0.5] de punto inicial obtenemos el siguiente minimo: ", E(w[0],w[1]))
print("Con las siguientes coordenadas: ",w,"\n")

initial_point = np.array([1,1])
w,it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print("Con [1,1] de punto inicial obtenemos el siguiente minimo: ", E(w[0],w[1]))
print("Con las siguientes coordenadas: ",w,"\n")

initial_point = np.array([2.1,-2.1])
w,it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print("Con [2.1,-2.1] de punto inicial obtenemos el siguiente minimo: ", E(w[0],w[1]))
print("Con las siguientes coordenadas: ",w,"\n")

initial_point = np.array([-3,3])
w,it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print("Con [-3,3] de punto inicial obtenemos el siguiente minimo: ", E(w[0],w[1]))
print("Con las siguientes coordenadas: ",w,"\n")

initial_point = np.array([-2,2])
w,it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print("Con [-2,2] de punto inicial obtenemos el siguiente minimo: ", E(w[0],w[1]))
print("Con las siguientes coordenadas: ",w,"\n")












###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa	
def pseudoinverse(?):
    #
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign(?) 

#Seguir haciendo el ejercicio...



