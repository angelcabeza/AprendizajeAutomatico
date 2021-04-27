# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Ángel Cabeza Martín
"""

import numpy as np
from sklearn import utils
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
            puntos_grafica.append(E(w[0],w[1]))
            iteraciones.append(iterations)
            
        iterations = iterations + 1
    return w, iterations    


learning_rate = 0.1
# Tope muy grande porque queremos que se pare cuando llegue a un error específico
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
# Este booleano nos servirá para saber qué función estamos usando
apartado3 = False

# Llamamos al algoritmo del gradiente descendiente con los siguientes argumentos
# initial_point -> Punto inicial desde el cual comenzaremos la búsqueda
# learning_rate -> Variable que indica el cambio entre iteración e iteración
# error2get -> Una de las opciones de parada del algoritmo
# maxIter -> Otra de las opciones de parada del algoritmo
w, it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print ( '¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor deE(u,v)inferior a 10−14? ', it)
print (' \n¿En qué coordenadas(u,v) se alcanzó por primera vez un valor igual o menor a 10−14\n (', w[0], ', ', w[1],')')
print ( 'El valor de la función en ese punto es: ', E(w[0],w[1]))

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


# En este bloque de instrucciones, ponemos el bool de apartado3 a truee porque vamos a 
# usar otra función, y tocamos los parámetros como se nos indican además pongo un error
# muy muy bajo para que eel algoritmo se pare cuando llegue a 50 iteraciones en vez 
# de por el error. Además creo dos listas que almacenarán los valores de la gráfica
# en los distintos puntos que va encontrando el algoritmo y otra lista que almacenará
# el número de iteraciones
apartado3 = True
learning_rate = 0.01
maxIter = 50
error2get = -999999
initial_point = np.array([-1.0,1.0])
puntos_grafica = []
iteraciones = []
w, it = gradient_descent(initial_point,learning_rate,error2get,maxIter)
print ( '\nEncontrado el mínimo en las coordenadas: (', w[0], ', ', w[1],')')


# Estas instrucciones son una manera de pasar de lista de python a array de numpy
puntos_funcion = np.array(puntos_grafica)
iterac = np.array(iteraciones)

# Instrucciones para pintar una gráfica 2D. El eje X corresponde a las iteraciones
# del algoritmo y el eje Y a los valores que va tomando en cada iteración
plt.plot(iterac,puntos_funcion)
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función')
plt.title('Gráfica que relaciona iteraciones y valor de la función (eta = 0.01)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora repetimos el mismo experimento cambiando el learning rate, las instrucciones
# son análogas al bloque de código anterior
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
plt.title('Gráfica que relaciona iteraciones y valor de la función (eta = 0.1) ')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# En este bloque de código vamos a asignar un learning_rate de 0.01 (en el apartado
# anterior hemos visto que es un learning_rate muy bueno para esta función) y un
# máximo de iteraciones 50 para distintos puntos iniciales y vamos a comprobar 
# a qué mínimo llegan en estas iteraciones.
print ('Vamos a aplicar el algoritmo del gradiente con distintos puntos iniciales\n')

initial_point = np.array([-0.5,-0.5])
learning_rate = 0.01

w,it = gradient_descent(initial_point,learning_rate,error2get,maxIter)

print("Con [-0.5,-0.5] de punto inicial obtenemos el siguiente minimo: ", E(w[0],w[1]))
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

input("\n--- Pulsar tecla para continuar ---\n")







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
    
    # He calculado el error según la fórmula dada en teoría (diapositiva 6)
    # Ein(w) = 1/N + SUM(wT*x - y)²
    #
    err = np.square(x.dot(w.T) - y)
    
    return err.mean()

def dErr(x,y,w):
    
    h_x = x.dot(w.T)
    
    dErr = h_x - y.T
    
    dErr = x.T.dot(dErr)
    
    dErr = (2 / x.shape[0]) * dErr
    
    return dErr.T

# Gradiente Descendente Estocastico
def sgd(x,y,learning_rate,num_batch,maxIter):

    # el tamaño de w será dependiendo del numero de columnas de x
    # shape[1] == columnas
    # shape[0] == filas
    w = np.zeros(x.shape[1])

    iterations = 1
    

    # en este caso solo tenemos de condicion las iteraciones
    while (iterations < maxIter ) :
        
        
        # Mezclamos x e y. Esta función solo cambia el orden de la matriz
        # el contenido no lo mezcla, es decir, si la columna 4 contiene los
        # valores 5 y 3, ahora puede que sea la columna 15 pero seguirá conteniendo
        # los mismos valores. Además lo hace en relación y para que aunque esté
        # todo mezclado a cada punto le siga correspondiendo su etiqueta
        utils.shuffle(x,y,random_state=1)
        
        
        # En este bucle vamos a crear tantos minibatchs como le hemos indicado
        # y vamos a aplicar la ecuación general para cada minibatch
        for i in range(0,num_batch):
        
            # Cogemos de x e y las filas que van desde i * tam_batch hasta i*tam_batch+tam_batch
            # p.ej si tam_batch = 64 cogeremos las filas 0-64, luego 64,128 y así
            minibatch_x = x[i*tam_batch:i*tam_batch+tam_batch]
            minibatch_y = y[i*tam_batch:i*tam_batch+tam_batch]            
        
            w = w - learning_rate*dErr(minibatch_x,minibatch_y,w)
        
        
        iterations = iterations + 1
        
    return w

# Pseudoinversa	
def pseudoinverse(x,y):
    
    # Calculamos las traspuestas de X e Y
    x_traspuesta = x.T
    y_traspuesta = y.T
    
    # Instrucciones para calcular la pseudoinversa de X
    x_pseudoinversa = x_traspuesta.dot(x)
    x_pseudoinversa = np.linalg.inv(x_pseudoinversa)
    x_pseudoinversa = x_pseudoinversa.dot(x_traspuesta)
    
    # Devolvemos el resultado de multiplicar la pseudoinversa de X por Y
    w = x_pseudoinversa.dot(y_traspuesta)
    
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

# Inicializamos los parámetros para el SGD
learning_rate = 0.01
tam_batch = 64
maxIter = 400

num_batch = int(len(x)/tam_batch)

x_aux = x.copy()
y_aux = y.copy()
w_sgd = sgd(x_aux,y_aux,learning_rate,num_batch,maxIter)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print("w: ",w_sgd)
print ("Ein: ", Err(x,y,w_sgd))
print ("Eout: ", Err(x_test, y_test, w_sgd))

# Separando etiquetas para poder escribir leyenda en el plot
etiq1 = []
etiq5 = []
for i in range(0,len(y)):
    if y[i] == 1:
        etiq5.append(x[i])
    else:
        etiq1.append(x[i])
        
etiq5 = np.array(etiq5)
etiq1 = np.array(etiq1)

# Plot de la separación de datos SGD

plt.scatter(etiq5[:,1],etiq5[:,2],c='red',label="5")
plt.scatter(etiq1[:,1],etiq1[:,2],c='blue',label="1")
plt.plot([0, 1], [-w_sgd[0]/w_sgd[2], (-w_sgd[0] - w_sgd[1])/w_sgd[2]],label="SGD")
plt.xlabel('Intensidad Promedio')
plt.ylabel('Simetria')
plt.legend()
plt.title('Modelo de regresión lineal obtenido con el SGD learning_rate = 0.01, 500 iteraciones')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


# BLoque de código para mostrar el modelo generado por la PSEUDOINVERSA
w_pseu = pseudoinverse(x,y)

print ('Bondad del resultado para alg pseudoinversa:\n')
print ("Ein: ", Err(x,y,w_pseu))
print ("Eout: ", Err(x_test, y_test, w_pseu))

# Plot de la separación de datos PSEUDOINVERSA

plt.scatter(etiq5[:,1],etiq5[:,2],c='red',label="5")
plt.scatter(etiq1[:,1],etiq1[:,2],c='blue',label="1")
plt.plot([0, 1], [-w_pseu[0]/w_pseu[2], (-w_pseu[0] - w_pseu[1])/w_pseu[2]],label="Pseudoinversa")
plt.xlabel('Intensidad Promedio')
plt.ylabel('Simetria')
plt.legend()
plt.title('Modelo de regresión lineal obtenido con la pseudoinversa')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

print("\nAhora vamos a ver gráficamente como se ajustan los alg fuera de la muestra")

# Separando etiquetas para poder escribir leyenda en el plot
etiq1 = []
etiq5 = []
for i in range(0,len(y_test)):
    if y_test[i] == 1:
        etiq5.append(x_test[i])
    else:
        etiq1.append(x_test[i])
        
etiq5 = np.array(etiq5)
etiq1 = np.array(etiq1)


# Plot de la separación de datos SGD

plt.scatter(etiq5[:,1],etiq5[:,2],c='red',label="5")
plt.scatter(etiq1[:,1],etiq1[:,2],c='blue',label="1")
plt.plot([0, 1], [-w_sgd[0]/w_sgd[2], (-w_sgd[0] - w_sgd[1])/w_sgd[2]],label="SGD")
plt.xlabel('Intensidad Promedio')
plt.ylabel('Simetria')
plt.legend()
plt.title('Modelo de regresión lineal fuera de la muestra obtenido con el SGD learning_rate = 0.01, 500 iteraciones')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


plt.scatter(etiq5[:,1],etiq5[:,2],c='red',label="5")
plt.scatter(etiq1[:,1],etiq1[:,2],c='blue',label="1")
plt.plot([0, 1], [-w_pseu[0]/w_pseu[2], (-w_pseu[0] - w_pseu[1])/w_pseu[2]],label="Pseudoinversa")
plt.xlabel('Intensidad Promedio')
plt.ylabel('Simetria')
plt.legend()
plt.title('Modelo de regresión lineal fuera de la muestra  obtenido con la pseudoinversa')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


##############################################################################
#
#               EJERCICIO 2
#
##############################################################################

print('Ejercicio 2\n')

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Devuelve un 1 si el signo es positivo y -1 si es negativo
def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign(np.square(x1-0.2) + np.square(x2) - 0.6)

# Función que genera índices aleatorios y al número con ese índice le cambia el signo
def ruido(etiquetas,porcentaje):
    num_etiquetas = len(etiquetas)
    
    etiquetas_a_cambiar = num_etiquetas * porcentaje
    
    etiquetas_a_cambiar = int(round(etiquetas_a_cambiar))
    
    for i in range (etiquetas_a_cambiar):
        indice = np.random.randint(0,1000)
        etiquetas[indice] = -etiquetas[indice]
        
    return etiquetas
          


# BLOQUE DE CÓDIGO PARA CALCULAR LOS PUNTOS SIN RUIDO Y SIN ETIQUETAS
print("Voy a generar un muestra de entrenamiento de 1000 puntos")
puntos_cuadrado = simula_unif(1000,2,1)

plt.scatter(puntos_cuadrado[:,0], puntos_cuadrado[:,1], c='b')
plt.title("Muestra de entrenamiento en el cuadrado [-1,1] x [-1,1]")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# BLOQUE DE CÓDIGO PARA DEFINIR LAS ETIQUETAS DE LOS PUNTOS
print("Ahora vamos a definir las etiquetas de la muestra")

etiqueta = []
for i in range(len(puntos_cuadrado)):
    etiqueta.append(f(puntos_cuadrado[i][0],puntos_cuadrado[i][1]))

etiquetas = np.array(etiqueta)


etiqueta_pos = []
etiqueta_neg = []

for i in range(len(etiquetas)):
    if (etiquetas[i] >= 0):
        etiqueta_pos.append(puntos_cuadrado[i])
    else:
        etiqueta_neg.append(puntos_cuadrado[i])
        
etiquetas_pos = np.array(etiqueta_pos)
etiquetas_neg = np.array(etiqueta_neg)
        
plt.scatter(etiquetas_pos[:,0],etiquetas_pos[:,1], c='yellow',label="f(x) >= 0")
plt.scatter(etiquetas_neg[:,0],etiquetas_neg[:,1], c='purple',label="f(x) < 0")
plt.title("Muestra de entrenamiento en el cuadrado [-1,1] x [-1,1], con las etiquetas sin ruido")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.legend()
plt.show()
        
    
input("\n--- Pulsar tecla para continuar ---\n")


# BLOQUE DE CÓDIGO EN EL QUE AÑADIMOS RUIDDO A LAS ETIQUETAS
print("A continuación introduciremos ruido sobre las etiquetas")


etiquetas = ruido(etiquetas,0.1)

etiqueta_pos = []
etiqueta_neg = []

for i in range(len(etiquetas)):
    if (etiquetas[i] >= 0):
        etiqueta_pos.append(puntos_cuadrado[i])
    else:
        etiqueta_neg.append(puntos_cuadrado[i])
        
etiquetas_pos = np.array(etiqueta_pos)
etiquetas_neg = np.array(etiqueta_neg)
        
plt.scatter(etiquetas_pos[:,0],etiquetas_pos[:,1], c='yellow',label="f(x) >= 0")
plt.scatter(etiquetas_neg[:,0],etiquetas_neg[:,1], c='purple',label="f(x) < 0")
plt.title("Muestra de entrenamiento en el cuadrado [-1,1] x [-1,1], con las etiquetas con ruido")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


#BLOQUE DE CÓDIGO PARA CALCULAR EL MODELO DE REGRESIÓN LINEAL DE LA MUESTRA
# EN ESTE BLOQUE HAY INSTRUCCIONES DIFICILES
print ("Finalmente vamos a calcular un modelo de regresión lineal con esta muestra")

caracteristicas = np.ones(puntos_cuadrado.shape[0])

# https://numpy.org/doc/stable/reference/generated/numpy.c_.html
# Esta función concatena los vectores por índices es decir características[i]
# lo concatena con puntos_cuadrado[i]
# los argumentos que se le pasa a esta función son los dos vectores que quieres
# concatenar
caracteristicas = np.c_[caracteristicas,puntos_cuadrado]


# Aquí mostramos las 10 primerass filas del vector de características
print("\nPrueba para ver si las características están construidas correctamente")
print(caracteristicas[: 10])

## Bloque de código  que llama al algoritmo de SGD y pinta el resultado
x = caracteristicas.copy()
y = etiquetas.copy()

num_batch = int(len(puntos_cuadrado)/tam_batch)

w = sgd(x,y,learning_rate,num_batch,maxIter)

print("\nW encontrada SGD = ", w)
print ('Bondad del resultado:\n')
print ("Ein: ", Err(caracteristicas,etiquetas,w))

plt.scatter(etiquetas_pos[:,0],etiquetas_pos[:,1], c='yellow',label="f(x) >= 0")
plt.scatter(etiquetas_neg[:,0],etiquetas_neg[:,1], c='purple',label="f(x) < 0")
plt.plot([-1, 1], [-w[0]/w[2], (-w[0] - w[1])/w[2]],label="SGD")
plt.title("Modelo de regresion lineal obtenido para la muestra de entrenamiento anterior")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.ylim(bottom = -1.1, top = 1.1)
plt.legend(loc="upper right")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Ahora vamos a repetir el proceso anterior 1000 veces pero con muestras distintas\n")
print("Paciencia esto puede tardar unos minutos...\n")

contador = 0
Ein = 0
Eout = 0

while(contador < 1000):
    contador += 1
    
    # Generamos la muestra de entrenamiento
    x = simula_unif(1000, 2, 1)
    
    #Generamos las etiquetas para la muestra de entrenamiento
    etiqueta = []
    for i in range(len(x)):
        etiqueta.append(f(x[i][0],x[i][1]))

    etiquetas = np.array(etiqueta)
    
    # Añadimos ruido
    y = ruido(etiquetas,0.1)
    
    #Creamos el vector de caracteristicas
    caracteristicas = np.ones(x.shape[0])
    caracteristicas = np.c_[caracteristicas,x]
    
    x_aux = caracteristicas.copy()
    y_aux = y.copy()
    
    num_batch = int(len(x)/tam_batch)
    
    w = sgd(x_aux,y_aux,learning_rate,num_batch,maxIter)
    
    #Variablee donde vamos a ir acumulando el error dentro de la muestra para calcular la media
    Ein += Err(caracteristicas,y,w)
    
    
    # Repetimos lo mismo para sacar Eout y evaluamos
    x_out = simula_unif(1000, 2, 1)
    
    #Generamos las etiquetas
    etiqueta = []
    for i in range(len(puntos_cuadrado)):
        etiqueta.append(f(x_out[i][0],x_out[i][1]))

    etiquetas = np.array(etiqueta)
    
    y_out = ruido(etiquetas,0.1)
    
    
    #Creamos el vector de caracteristicas
    caracteristicas_out = np.ones(x_out.shape[0])
    caracteristicas_out = np.c_[caracteristicas_out,x_out]
    
    Eout += Err(caracteristicas_out,y_out,w)
    

print("Valor medio Ein: ",Ein/1000)
print("\nValor medio de Eout: ", Eout/1000)


input("\n--- Pulsar tecla para continuar ---\n")

#COMIENZA EL SEGUNDO PUNTO DEL EJERCICOI 2

#Bloque de código para generar los puntos de la muestra de entrenamiento y pintar la gráfica
print("Vamos a repetir el experimento anterior pero con características no lineales")

x = simula_unif(1000, 2, 1)

plt.scatter(x[:,0], x[:,1], c='b')
plt.title("Muestra de entrenamiento en el cuadrado [-1,1] x [-1,1]")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Bloque de código para generar sus etiquetas y pintar la gráfica
print("Ahora vamos a definir las etiquetas de la muestra")

etiqueta = []
for i in range(len(puntos_cuadrado)):
    etiqueta.append(f(x[i][0],x[i][1]))

etiquetas = np.array(etiqueta)


etiqueta_pos = []
etiqueta_neg = []

for i in range(len(etiquetas)):
    if (etiquetas[i] >= 0):
        etiqueta_pos.append(x[i])
    else:
        etiqueta_neg.append(x[i])
        
etiquetas_pos = np.array(etiqueta_pos)
etiquetas_neg = np.array(etiqueta_neg)
        
plt.scatter(etiquetas_pos[:,0],etiquetas_pos[:,1], c='yellow',label="f(x) >= 0")
plt.scatter(etiquetas_neg[:,0],etiquetas_neg[:,1], c='purple',label="f(x) < 0")
plt.title("Muestra de entrenamiento en el cuadrado [-1,1] x [-1,1], con las etiquetas sin ruido")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.legend()
plt.show()
        
    
input("\n--- Pulsar tecla para continuar ---\n")

#Bloque de código para añadir ruido a las etiquetas y pintar de nuevo la gráfica
print("A continuación introduciremos ruido sobre las etiquetas")


etiquetas = ruido(etiquetas,0.1)

etiqueta_pos = []
etiqueta_neg = []

for i in range(len(etiquetas)):
    if (etiquetas[i] >= 0):
        etiqueta_pos.append(x[i])
    else:
        etiqueta_neg.append(x[i])
        
etiquetas_pos = np.array(etiqueta_pos)
etiquetas_neg = np.array(etiqueta_neg)
        
plt.scatter(etiquetas_pos[:,0],etiquetas_pos[:,1], c='yellow',label="f(x) >= 0")
plt.scatter(etiquetas_neg[:,0],etiquetas_neg[:,1], c='purple',label="f(x) < 0")
plt.title("Muestra de entrenamiento en el cuadrado [-1,1] x [-1,1], con las etiquetas con ruido")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Bloque de código para llamar al algoritmo y pintar la elipse
print ("Finalmente vamos a calcular un modelo de regresión con esta muestra")

caracteristicas = np.ones(x.shape[0])

# https://numpy.org/doc/stable/reference/generated/numpy.c_.html
caracteristicas = np.c_[caracteristicas,x[:, 0], x[:, 1],x[:, 0]*x[:, 1],  np.square(x[:, 0]),  np.square(x[:, 1])]

print("\nPrueba para ver si las características están construidas correctamente")
print(caracteristicas[: 10])

x_aux = caracteristicas.copy()
y_aux = etiquetas.copy()

num_batch = int(len(x)/tam_batch)

w = sgd(x_aux,y_aux,learning_rate,num_batch,maxIter)

print("\nW encontrada = ", w)
print ('Bondad del resultado:\n')
print ("Ein: ", Err(caracteristicas,etiquetas,w))


plt.scatter(etiquetas_pos[:,0],etiquetas_pos[:,1], c='yellow',label="f(x) >= 0")
plt.scatter(etiquetas_neg[:,0],etiquetas_neg[:,1], c='purple',label="f(x) < 0")

# PARA PINTAR LA ELIPSE HE CREADO PUNTOS DESDE -1 A 1 de 0.025 EN 0.025 Y HE IDO SUSTITUYENDO
# EN LA FUNCIÓN DE UNA ELIPSE PARA DIBUJARLA CON CONTOUR
x_range = np.arange(-1,1,0.025)
y_range = np.arange(-1,1,0.025)
valor_x, valor_y = np.meshgrid(x_range,y_range) 
func = w[0] + valor_x*w[1] + valor_y*w[2] + valor_x*valor_y*w[3] + ((valor_x)**2)*w[4] + ((valor_y)**2)*w[5]

#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
# Los dos primeros argumentos (X,Y) son los vectores con los valores de los puntos
# el tercer argumento (Z) es la funcion que evalua los puntos
# y el cuarto argumento es el nivel del contorno (en este caso el nivel 0)
# si le indicaramos por ejemplo un array [0,1] dibujaría dos contornos con los
# mismos puntos solo que uno más grande que otro
plt.contour(valor_x,valor_y,func,0)


plt.title("Modelo de regresion lineal obtenido para la muestra de entrenamiento anterior")
plt.xlabel('Valor de x1')
plt.ylabel('Valor de x2')
plt.ylim(bottom = -1.1, top = 1.1)
plt.legend(loc="upper right")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Bloque de código para repetir el proceso anterior 1000 veces y calcular la 
# media de EIN y EOUT

print("Ahora vamos a repetir el proceso anterior 1000 veces pero con muestras distintas\n")
print("Paciencia esto puede tardar unos minutos...\n")

contador = 0
Ein = 0
Eout = 0
while(contador < 1000):
    contador += 1
    
    x = simula_unif(1000, 2, 1)
    
    #Generamos las etiquetas
    etiqueta = []
    for i in range(len(x)):
        etiqueta.append(f(x[i][0],x[i][1]))

    etiquetas = np.array(etiqueta)
    
    y = ruido(etiquetas,0.1)
    
    
    #Creamos el vector de caracteristicas
    caracteristicas = np.ones(x.shape[0])
    caracteristicas = np.c_[caracteristicas,x[:, 0], x[:, 1],x[:, 0]*x[:, 1],  np.square(x[:, 0]),  np.square(x[:, 1])]
    
    x_aux = caracteristicas.copy()
    y_aux = y.copy()
    
    num_batch = int(len(x)/tam_batch)
    
    w = sgd(x_aux,y_aux,learning_rate,num_batch,maxIter)
    
    #Variable donde vamos a ir acumulando el error dentro de la muestra para calcular la media
    Ein += Err(caracteristicas,y,w)
    
    
    # Repetimos lo mismo para sacar Eout y evaluamos
    x_out = simula_unif(1000, 2, 1)
    
    #Generamos las etiquetas
    etiqueta = []
    for i in range(len(puntos_cuadrado)):
        etiqueta.append(f(x_out[i][0],x_out[i][1]))

    etiquetas = np.array(etiqueta)
    
    y_out = ruido(etiquetas,0.1)
    
    
    #Creamos el vector de caracteristicas
    caracteristicas_out = np.ones(x_out.shape[0])
    caracteristicas_out = np.c_[caracteristicas_out,x_out[:, 0], x_out[:, 1],x_out[:, 0]*x_out[:, 1],  np.square(x_out[:, 0]),  np.square(x_out[:, 1])]
    
    Eout += Err(caracteristicas_out,y_out,w)
    

print("Valor medio Ein: ",Ein/1000)
print("\nValor medio de Eout: ", Eout/1000)