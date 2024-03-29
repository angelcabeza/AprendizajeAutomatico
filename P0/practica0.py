#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:40:45 2021

@author: Angel Cabeza Martín Grupo 3
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import pi
import math as mt

##############################################################################
# INFORMACIÓN RECOGIDA DE:
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html 

# Con esta instrucción leemos la base de datos iris de scikit learn
iris = datasets.load_iris()

# Recogemos las características del dataset
x = iris.data
# Recogemos las clases del dataset
y = iris.target

print("Estas son las características del dataset")
print(x)


print ("\nEstas son las clases del dataset")
print(y)

##############################################################################



# Con estas dos instrucciones recogemos de la matriz de características
# con ":" indicamos que queremos todas las filas
# y con el segundo argumento indicamos la columna que deseeamos
col_1_x = iris.data[:, 0]
col_3_x = iris.data[:, 2]

#Vamos a hacer el array de colores según la clase 
# 0 -> Naranja (orange)
# 1 -> Negro (black)
# 2 -> Verde (green)
color = []

for i in range(len(y)):
    if y[i] == 0:
        color.append('orange')
    if y[i] == 1:
        color.append('black')
    if y[i] == 2:
        color.append('green')

##############################################################################
# INFORMACIÓN RECOGIDA DE:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
##############################################################################

# Indicamos los valores de x (col_1_x) y de y (col_3_x) del gráfico y le pasamos
# el array de colores que le correspodne a cada punto
plt.scatter(col_1_x,col_3_x,c=color)

##############################################################################
# INFORMACIÓN RECOGIDA DE:
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
##############################################################################

# Con esta instrucción personalizamos la leyenda, le estamos diciendo que pinte
# un punto (marker = '.') de color naranja (markerfacecolor='orange') y con
# un tamaño de 15 (markersize=15)
legends_elements = [plt.Line2D([0], [0], marker='.', color='w', label='Clase 0',
                          markerfacecolor='orange', markersize=15),
                     plt.Line2D([0], [0], marker='.', color='w', label='Clase 1',
                           markerfacecolor='black', markersize=15),
                     plt.Line2D([0], [0], marker='.', color='w', label='Clase 2',
                           markerfacecolor='green', markersize=15)]

# Esta instrucción pone la leyenda usando la lista que hemos creado antes.
# Además le he indicado que salga arriba a la izquierda
plt.legend(handles=legends_elements,loc="upper left")
    
# Con esta instrucción se muestra el gráfico                
plt.show()

input("Pulse una tecla para continuar")

###############################################################################
## COMIENZO DEL EJERCICIO 2
###############################################################################

##############################################################################
# INFORMACIÓN RECOGIDA DE:
# https://realpython.com/train-test-split-python-data/
# https://es.wikipedia.org/wiki/Muestreo_estratificado
##############################################################################

print("\n\n Voy a separar los datos del dataset iris en dos partes (training y test)...\n\n")

training = []
test = []

# Concatenamos los arrays con las características 1 y 3 para separarlos
# en training y test
col1y3 = col_1_x + col_3_x

# En este ejercicio he entendido que hay que separar las características 1 y 3
# si hubiera que separar todo x lo único que habría que hacer es cambiar el primer
# argumento por el vector que contiene todos los datos (x)
 
# Con el parámetro startify dividiimos x en grupos homogéneos según y 
# con los parametros de size indicamos el % del array a dividir que queremos
# destinar a train y a test
training,test = train_test_split(col1y3,train_size=0.75,test_size=0.25,stratify=y)


print("He dividido los datos del dataset iris en dos arrays\n Training con un 75% de los datos y test con un 25\n")
     
print("Este es el array de training:\n ", training)
print("Su tamaño es de: ",len(training)," que es el 75% de 150 elementos\n")

print("Este es el array de test:\n ", test)
print("\nSu tamaño es de: ", len(test), " que es el 25% de 150 elementos\n")

input("Pulse una tecla para continuar")

##############################################################################
# COMIENZO DEL EJERCICIO 3
##############################################################################

valores = []
senos = []
cosenos = []
tangentes = []

# linspace es un método que genera valores aleatorios equispaciados
# Sus argumentos son: start: valor incial, end: valor final
# el número de números que va a generar en este intervalo
valores = np.linspace(0,4*pi,100)


print("Estos son los valores elegidos: \n",valores)

# En este bucle calculamos el seno, el coseno, y la tengente de la suma de los
# senos y cosenos
for i in range(len(valores)):
    senos.append(mt.sin(valores[i]))
    cosenos.append(mt.cos(valores[i]))
    tangentes.append(mt.tanh(senos[i] + cosenos[i]))

    
# Instrucciones para pintar la gráfica

##############################################################################
# INFORMACIÓN RECOGIDA DE:
# DIAPOSITIVA 49 DEL PDF LLAMADO CLASE 2 PRORPORCIONADO EN PRÁCTICAS
##############################################################################

# Hacemos un plot del vector, el segundo argumento indica el color y la forma
# de representar los datos ("--" indica que sea una línea discontinua)
# y el tercer argumento indica la etiqueta que le ponemos a esta línea
plt.plot(senos,'g--',label='seno')
plt.plot(cosenos,'k--',label='coseno')
plt.plot(tangentes,'r--',label='tangente hiperbólica')


# Instrucciones para poner nombre al eje X y al eje Y y para poner título a la
# gráfica 
plt.xlabel('X axis')
plt.ylabel('Y axis')

# Le ponemos titulo a la gráfica
plt.title('Funciones seno coseno y tangente de la suma de los senos y cosenos')
plt.legend(loc="lower left")
plt.show()

##############################################################################