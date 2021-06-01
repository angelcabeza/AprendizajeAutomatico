#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:27:43 2021

@author: angel
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lars
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Inicializamos la semilla
np.random.seed(400)

input("\n--- Pulsar tecla para continuar ---\n")

# Leo el conjunto de datos
print("Leyendo el conjunto de datos..")
data = np.genfromtxt('./datos/train.csv',delimiter=',')

#quitamos la primera fila porque contenia el nombre de cada característica
data = np.delete(data,0,axis=0)


# Vamos a extraer los vectores de características y nuestra Y del problema
x = data[:, :-1]
y = np.ravel(data[:, -1:])

input("\n--- Pulsar tecla para continuar ---\n")

# Vamos a dividir el conjunto en train y test

x_train,x_test,y_train, y_test = train_test_split(x,y,train_size=0.75,test_size=0.25)

print("Tamaño del conjunto de train: ", len(x_train))

print("\nTamaño del conjunto de test: ", len(x_test))

input("\n--- Pulsar tecla para continuar ---\n")

# Comprobamos si hay valores perdidos
print("Comprobando si hay valores perdidos")

print("¿Existen valores perdidos?: ", end='')
print(pd.DataFrame(np.vstack([x_train, x_test])).isnull().values.any())

input("\n--- Pulsar tecla para continuar ---\n")

print("Lista de las columnas con un solo valor en todas las muestras")

# Aquí estamos mirando si hay columnas con un solo valor
for i in range(x_train.shape[1]):
    long = len(np.unique(x_train[:,i]))
    
    if long == 1:
        print(i, )

print("Ahora vamos a ver si hay columnas con pocos valores únicos")

# Aqui miramos si hay columnas con pocos valores únicos 
for i in range(x_train.shape[1]):
    print(i, len(np.unique(x_train[:,i])))


input("\n--- Pulsar tecla para continuar ---\n")

print("Mostrando características con un coeficiente de correlación de Pearson > 0.9 o < -0.9")

# Calculamos el coeficiente Pearson para cada columna
p_corr = np.corrcoef(x_train,rowvar=False)

# Mostramos las columnas con una correlación > 0.9 o < -0.9
for a in range(len(p_corr)):
    for b in range(len(p_corr[a])):
        if ((a != b) and (p_corr[a,b] > 0.9 or p_corr[a,b] < -0.9)):
            print("{} con {} correlación: {}".format(a,b,p_corr[a,b]))
            
input("\n--- Pulsar tecla para continuar ---\n")

print("Estandarizando los datos...")

# Estandarización de los datos
x_train = StandardScaler(copy=False).fit_transform((x_train))
x_test = StandardScaler(copy=False).fit_transform((x_test))

print("¡Estandarizacion realizada!")

input("\n--- Pulsar tecla para continuar ---\n")

# Aqui aplicamos PCA, el mismo para train y para test
print("Aplicando PCA...")

pca = PCA(0.95)

pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

print("Nuevas dimensiones de x_train: " , x_train.shape[1])

input("\n--- Pulsar tecla para continuar ---\n")

# Estos son parámetros que nos hacen falta para llamar a GridSearchCV
parametros = [{'penalty': ['l1','l2'], 'alpha': np.logspace(-20,4,25), 'eta0': np.logspace(-4,0,5)}]
scoring = 'neg_mean_absolute_error'
col = ['mean_fit_time', 'mean_test_score', 'std_test_score', 'rank_test_score']


# BLOQUE DE CÓDIGO DONDE ENTRENO LOS MODELOS CON CV
# La linea de código comentada es para hacer el grid de parámetros, la he comentado
# para que al corregir no tarde tanto (tarda como unos 10 minutos) y justo debajo
# entreno el mismo modelo con los mejores parámetros que nos salen de hacer el grid
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #sgdReg = GridSearchCV(SGDRegressor(random_state=400),parametros,scoring)
    sgdReg = GridSearchCV(SGDRegressor(random_state=400),{'alpha':[0.1],'eta0':[0.0001],'penalty':['l2']},scoring)
    sgdReg.fit(x_train,y_train)
    linReg = GridSearchCV(LinearRegression(),{'normalize': [False]},scoring)
    linReg.fit(x_train,y_train)
    lars = GridSearchCV(Lars(),{'normalize': [False]},scoring)
    lars.fit(x_train,y_train)

# Pinto el MAE usando CV
print("Resultados de selección de hiperparámetros por CV:")
print("SGD mejores parámetros: ",sgdReg.best_params_)
print("SGD CV-MAE: ", -sgdReg.best_score_)
print("LARS CV-MAE", -lars.best_score_)
print("Pseudoinversa CV-MAE: ", -linReg.best_score_)

# Calculo e imprimo Ein 
sgdReg = SGDRegressor(random_state=400,alpha=0.1,eta0=0.0001,penalty='l2')
sgdReg.fit(x_train,y_train)
sgd_pred = sgdReg.predict(x_train)
ein = mean_absolute_error(y_train,sgd_pred)
print("Ein: ", ein)


input("\n--- Pulsar tecla para continuar ---\n")

# Saco Etest del mejor modelo usando MAE
sgd_pred = sgdReg.predict(x_test)
etest_lin = mean_absolute_error(y_test,sgd_pred)

print("SGD Test-MAE ", etest_lin)

input("\n--- Pulsar tecla para continuar ---\n")

# Aquí saco las cotas usadno la desigualdad de Hoeffding
eout_lin = ein + np.sqrt((1/(2*len(x_train))) * np.log(2/0.05))

print("Cota desigualdad de Hoeffding usando Ein SGD Lin: ", eout_lin)

eout_lin = etest_lin + np.sqrt((1/(2*len(x_test))) * np.log(2/0.05))

print("Cota desigualdad de Hoeffding usando Etest SGD Lin: ",eout_lin)

input("\n--- Pulsar tecla para continuar ---\n")

# Aquí calculo las cotas de Eout del mejor modelo entrenado con todos los datos
# Pero primero a x le aplico el MISMO preprocesado de datos que a xtrain
x = StandardScaler(copy=False).fit_transform((x))
x = pca.transform(x)


sgdReg = SGDRegressor(alpha=0.1,eta0=0.0001,penalty='l2',random_state=400)
sgdReg.fit(x,y)

sgd_pre = sgdReg.predict(x)
ein = mean_absolute_error(y,sgd_pre)
eoutlin = ein + np.sqrt((1/(2*len(x))) * np.log(2/0.05))
                            
print("Ein SGD Lin: ", ein)
print("Cota Eout usando Ein y desigualdad de Hoeffding SGD_Lin: ",eoutlin)


input("\n--- Pulsar tecla para continuar ---\n")