#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:40:45 2021

@author: angel
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

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