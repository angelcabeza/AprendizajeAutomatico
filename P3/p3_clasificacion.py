#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:37:04 2021

@author: angel
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC



def readData(path,delim=' ', dtype=np.float64):
    data_set = np.loadtxt(path,dtype,None,delim)
    
    x = data_set[:, :-1]
    y = np.ravel(data_set[:, -1:])
    
    return x,y

# Fijo la semilla
np.random.seed(1)

# Lectura de los datos
print("Voy a leer los datos")
x, y = readData('./datos/Sensorless_drive_diagnosis.txt')

# BLOQUE DE CÓDIGO PARA VER SI EL DATASET ESTÁ BALANCEADO
unique,contador = np.unique(y, return_counts=True)

print(contador.shape)
plt.title("Recuento de los valores del conjunto de datos")
plt.ylabel("Numero de elementos en cada clase")
plt.xlabel("Clase")
plt.bar(unique,contador,width=0.4,tick_label=["1","2","3","4","5","6","7","8","9","10","11"])
plt.show()
################################################################################

input("\n--- Pulsar tecla para continuar ---\n")

# BLOQUE DE CÓDIGO PARA VER SI LOS CONJUNTOS SIGUEN BALANCEADOS DESPUES DE SEPARARLOS EN TRAIN Y TEST
#################################################################################################
print("Vamos a separar el conjunto de datos en training y test")

x_train,x_test,y_train, y_test = train_test_split(x,y,train_size=0.68,test_size=0.32,stratify=y)


unique,contador = np.unique(y_train, return_counts=True)

print(contador.shape)
plt.title("Recuento de los valores del conjunto de datos (training)")
plt.ylabel("Numero de elementos en cada clase")
plt.xlabel("Clase")
plt.bar(unique,contador,width=0.4,tick_label=["1","2","3","4","5","6","7","8","9","10","11"])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
unique,contador = np.unique(y_test, return_counts=True)

print(contador.shape)
plt.title("Recuento de los valores del conjunto de datos (test)")
plt.ylabel("Numero de elementos en cada clase")
plt.xlabel("Clase")
plt.bar(unique,contador,width=0.4,tick_label=["1","2","3","4","5","6","7","8","9","10","11"])
plt.show()
################################################################################################

################################################################################################
# PREPROCESAMIENTO DE DATOS
input("\n--- Pulsar tecla para continuar ---\n")

# Miramos si hay valores perdidos
print("¿Existen valores perdidos?: ", end='')
print(pd.DataFrame(np.vstack([x_train, x_test])).isnull().values.any())

input("\n--- Pulsar tecla para continuar ---\n")

# Miramos si hay columnas con un solo valro en todas sus muestras
print("Lista de las columnas con un solo valor en todas las muestras")

for i in range(x_train.shape[1]):
    long = len(np.unique(x_train[:,i]))
    
    if long == 1:
        print(i, )

# Miramos si hay columnas con pocos valores unicos
print("Ahora vamos a ver si hay columnas con pocos valores únicos")

for i in range(x_train.shape[1]):
    print(i, len(np.unique(x_train[:,i])))
    
input("\n--- Pulsar tecla para continuar ---\n")

# Miramos si hay filas duplicadas
print("¿Hay filas duplicadas?")
dups = pd.DataFrame(x).duplicated()

print(dups.any())

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

print("Estos son los valores máximos y mínimos de cada columna")

maximos = np.max(x_train,axis=0);
minimos = np.min(x_train, axis=0)

for i in range(len(maximos)):
    print("Col {} max = {} min = {}".format(i,maximos[i],minimos[i]))
    
    
input("\n--- Pulsar tecla para continuar ---\n")
# BLoque de código donde estandarizamos los datoss
print("Estandarizando los datos...")

x_train = StandardScaler(copy=False).fit_transform((x_train))
x_test = StandardScaler(copy=False).fit_transform(x_test)

print("¡Estandarizacion realizada!")

################################################################################
input("\n--- Pulsar tecla para continuar ---\n")


# Estos son los parámetros con los que vamos a probar y hacer el grid de parámetros
parametros_LR = [{'C': np.logspace(-3,3,7)}]
parametros_PT = [{'tol': np.logspace(-2,2,5),'eta0': [0.001,0.1,1,5,10]}]
parametros_SVM = [{'C': np.logspace(-3,-1,3)}]

columns_log = ['mean_fit_time', 'param_C', 'mean_test_score',
               'std_test_score', 'rank_test_score']

columns_pt = ['mean_fit_time', 'param_tol','param_eta0', 'mean_test_score',
               'std_test_score', 'rank_test_score']

columns_svm = ['mean_fit_time', 'param_C', 'mean_test_score',
               'std_test_score', 'rank_test_score']


# EN ESTE BLOQUE DE CÓDIGO HAGO LOS GRID DE PARÁMETROS PERO LOS HE COMENTADO PARA REDUCIR
# EL TIEMPO DE EJECUCIÓN (TARDABA UNOS 15 MINUTOS) Y DEBAJO ENTRENO EL MODELO CON LOS MEJORES PARÁMETROS
# =============================================================================
#with warnings.catch_warnings():    
#    warnings.simplefilter("ignore")
#    logReg = GridSearchCV(LogisticRegression(penalty='l1',solver='saga',multi_class='multinomial',random_state=1),parametros_LR)
#    logReg.fit(x_train,y_train)
#    print('Cross Validation para Regresión Logística\n', pd.DataFrame(logReg.cv_results_,columns=columns_log).to_string())
#    perceptron = GridSearchCV(Perceptron(penalty='l1',random_state=1),parametros_PT)
#    perceptron.fit(x_train,y_train)
#    print('Cross Validation para Perceptron\n', pd.DataFrame(perceptron.cv_results_, columns=columns_pt).to_string())
#    svm = SVC(kernel='linear',probability=True,decision_function_shape='ovr',random_state=(1))
#    svm.fit(x_train,y_train)
     
#print("\nParametros seleccionados para RL usando validacion cruzada")
#print(logReg.best_params_)
#print("\nCross Validation para LR: ", logReg.best_score_)

#print("\nParametros seleccionados para Perceptron usando validacion cruzada")
#print(perceptron.best_params_)
#print("\nCross Validation para Perceptron: ", perceptron.best_score_)



# BLoque de código para entrenar los modelos y sacar CV-Accuracy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logReg = GridSearchCV(LogisticRegression(penalty='l1',solver='saga',multi_class='multinomial',random_state=1),{'C':[0.1]})
    logReg.fit(x_train,y_train)
    print("\nCross Validation para LogReg: ", logReg.best_score_)
       
    perceptron = GridSearchCV(Perceptron(penalty='l1',random_state=1),{'eta0':[0.001],'tol':[0.01]})
    perceptron.fit(x_train,y_train)
    print("\nCross Validation para Perceptron: ", perceptron.best_score_)
       
    svm = SVC(kernel='linear',probability=True,decision_function_shape='ovr',random_state=(1))
    svm.fit(x_train,y_train)
    
cvsvm =  np.mean(cross_val_score(svm,x_train,y_train));
print("\nCross Validation para SVMLineal: ", cvsvm)


input("\n--- Pulsar tecla para continuar ---\n")

# BLoque dee código para calcular las cotas de Eout usando la desigualdad de Hoeffding
print("Cota Eout usando CV para SVM Lineal", 1 - cvsvm)

ein = 1 - svm.score(x_train,y_train)
print("Ein de SVM Lineal: ", ein)

etest = 1 - svm.score(x_test,y_test)
print("Etest para SVM Lineal", etest)

eout = etest + np.sqrt((1/(2*len(x_test))) * np.log(2/0.05))
print("Cota de Eout usando Etest: ", eout)

eout = ein + np.sqrt((1/(2*len(x_train))) * np.log(2/0.05))
print("Cota de Eout usando Ein: ", eout)

# TARDA UNOS 25 MINUTOS
input("\n--- Pulsar tecla para continuar ---\n")

# Bloque de código para entrenar al modelo con todos los datos y sacar la cota de Eout
# le aplicamos EL MISMO PREPROCESAMIENTO a x que a xtrain 
x = StandardScaler(copy=False).fit_transform((x))
svmALL = SVC(kernel='linear',probability=True,decision_function_shape='ovr',random_state=(1))
svmALL.fit(x,y)

ein = 1 - svmALL.score(x,y)
print("Ein para SVM Lineeal: ", ein)

eout = ein + np.sqrt((1/(2*len(x))) * np.log(2/0.05))
print("\nCota de Eout usando Ein y desigualdad de Hoeffding: ", eout)

input("\n--- Pulsar tecla para continuar ---\n")
