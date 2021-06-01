# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Ángel Cabeza Martín
"""
import numpy as np
import matplotlib.pyplot as plt



# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# Esta función calcula el porcentaje de puntos mal etiquetados
def num_errores(x,y,a,b):
    
    # contador de errores a 0
    errores = 0
    
    for i in range(len(x)):
        # si el signo del punto respecto de la recta es distinto al de la etiqueta real
        #aumentamos el contador de errores
        if signo(f(x[i][0],x[i][1],a,b)) != y[i]:
            errores += 1
            
    errores = errores/len(x)
    
    return errores


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

# Sacamos de manera uniforme los 50 puntos
x = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE

# BLOQUE DE CÓDIGO PARA DIBUJAR LA NUBE DE PUNTOS
plt.scatter(x[:,0],x[:, 1], c='orange')
plt.title("Nube de 50 puntos simula_unif dimensión 2 rango -50,50")
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()
plt.clf()
################################################

input("\n--- Pulsar tecla para continuar ---\n")

# Sacamos los 50 puntos usando una distribución gausiana
x = simula_gaus(50, 2, np.array([5,7]))

# BLOQUE DE CÓDIGO PARA DIBUJAR LA NUBE DE PUNTOS
plt.scatter(x[:,0],x[:,1],c='orange')
plt.title("Nube de 50 puntos simula_gaus dimensión 2 sigma [5,7]")
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()
plt.clf()
################################################


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE

# Sacamos 100 puntos de manera uniforme
x = simula_unif(100,2,[-50,50])

# Esto nos va a servir para otro ejercicio en el que tenemos que reproducir estos resultados
x_ap3 = x.copy();

# Sacamos el término independiente y la pendiente de una recta de manera aleatoria
a,b = simula_recta([-50,50])

# Esto nos va a servir para otro ejercicio en el que tenemos que reproducir estos resultados
a_ap3 = a.copy()
b_ap3 = b.copy()

# Bloque de código para etiquetar los puntos
etiquetas = []
posibles_etiquetas = (1,-1)
colores = {1:'orange',-1: 'black'}

#etiquetamos cada punto según la función de la recta
for punto in x:
    etiquetas.append(f(punto[0],punto[1],a,b))
    
# dibujamos los puntos sin etiquetar
plt.scatter(x[:,0],x[:,1],c='green')
plt.title("Nube de 100 puntos bidimensionales en el intervalo -50,50")
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

# Bloque de código donde dibujamos los puntos etiquetados, cada uno de un color
for etiqueta in posibles_etiquetas:
        indice = np.where(np.array(etiquetas) == etiqueta)
        
        plt.scatter(x[indice,0], x[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))
        
        
plt.plot([-50,50],[a*-50 +b, a*50 +b], 'k-', label='Recta aleatoria')

plt.title("Nube de 100 puntos bidimensionales en el intervalo [-50,50], etiquetados según una recta")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()

##############################################################################

print ("Porcentaje de errores en el etiquetado: " + str(num_errores(x,etiquetas,a,b)))
input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE

# Esta función calcula el porcentaje de puntos mal etiquetados
def num_errores_func(x,y,func):
    
    errores = 0
    val = func(x)
    for i in range(len(x)):
        #Si el signo de la función que le pasamos por argumento es distinto
        # a la etiqueta real
        if signo(val[i]) != y[i]:
            errores += 1
            
    errores = errores/len(x)
    
    return errores


# Función para aplicar el ruido
def ruido(etiquetas):
    # Sacamos los índices de los puntos con etiquetas positivas
    indices_pos = np.where(np.array(etiquetas) == 1)
    indices_pos = indices_pos[0]
    
    # Sacamos los índices de los puntos con etiquetas negativas
    indices_neg = np.where(np.array(etiquetas) == -1)
    indices_neg = indices_neg[0]
    
    #aplicamos el ruido a las etiquetas positivas
    ruido_aplicar = len(indices_pos) * 0.1
    ruido_aplicar = int(round(ruido_aplicar))
    indices = np.random.choice(indices_pos,ruido_aplicar,replace=False)
    
    for i in indices:
        etiquetas[i] = -etiquetas[i]
        
    #aplicamos el ruido a las etiquetas negativas
    ruido_aplicar = len(indices_neg) * 0.1
    ruido_aplicar = int(round(ruido_aplicar))
    indices = np.random.choice(indices_neg,ruido_aplicar,replace=False)
    
    for i in indices:
        etiquetas[i] = -etiquetas[i]
        
    return etiquetas

# Al vector de etiquetas del apartado anterior le aplicamos el ruido
etiquetas = ruido(etiquetas)

# Esto lo usaremos para otro ejercicoi
etiquetas_ap3b = etiquetas.copy()
    
# dibujamos los puntos etiquetados, cada uno de un color
for etiqueta in posibles_etiquetas:
        indice = np.where(np.array(etiquetas) == etiqueta)
        
        plt.scatter(x[indice,0], x[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))
        
  
# Bloque de código para pintar los datos
plt.plot([-50,50],[a*-50 +b, a*50 +b], 'k-', label='Recta aleatoria')

plt.title("Nube de 100 puntos bidimensionales en el intervalo {-50} {50}, etiquetados según una recta con ruido")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()
#########################################################################

# Mostramos el porcentaje de error
print ("Porcentaje de errores en el etiquetado: " + str(num_errores(x,etiquetas,a,b)))

input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
 
# Estas son las distintas funciones que nos pide el apartado
def f1(x):
    return np.float64((x[:, 0]-10)**2 + (x[:, 1] - 20)**2 -400)
    
def f2(x):
    return np.float64(0.5*(x[:, 0] + 10)**2 + (x[:, 1] - 20)**2 -400)

def f3(x):
    return np.float64(0.5*(x[:,0] - 10)**2 - (x[:,1] + 20)**2 -400)

def f4(x):
    return np.float64(x[:,1] - (20*(x[:,0]**2)) - (5*x[:,0]) + 3)
    
########################################################################

#CODIGO DEL ESTUDIANTE
etiquetas_3 = etiquetas.copy()

# EN ESTE BLOQUE DE CÓDIGO VAMOS A PINTAR LOS DATOS USANDO LA FUNCIÓN PLOT DATOS CUAD
# Y EL PORCENTAJE DE FALLOS
plot_datos_cuad(x,etiquetas_3,f1,"Funcion f1 con etiquetas de la recta")

print ("Porcentaje de errores en el etiquetado de f1: " + str(num_errores_func(x,etiquetas,f1)))

input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,etiquetas_3,f2,"Funcion f2 con etiquetas de la recta")

print ("Porcentaje de errores en el etiquetado de f2: " + str(num_errores_func(x,etiquetas,f2)))

input("\n--- Pulsar tecla para continuar ---\n")


plot_datos_cuad(x,etiquetas_3,f3,"Funcion f3 con etiquetas de la recta")

print ("Porcentaje de errores en el etiquetado de f3: " + str(num_errores_func(x,etiquetas,f3)))
input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,etiquetas_3,f4,"Funcion f4 con etiquetas de la recta")

print ("Porcentaje de errores en el etiquetado de f4 : " + str(num_errores_func(x,etiquetas,f4)))

###########################################################################################
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# ALGORITMO PERCEPTRON
def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    
    #inicializo w al valor incial
    w = vini.copy()
    # Variables de cotnrol para saber cuándo dejar de iterar
    itera = 0;
    hay_mejora = True
    
    
    # Mientras mejore w o y las iteraciones sean menores que el máximo
    while hay_mejora and itera < max_iter:
        # asumimos que no va a haber mejora
        hay_mejora  = False
        
        # Recorremos los datos
        for i in range(0, len(datos)):
            # Calculamos el signo del dato
            valor = signo(w.T.dot(datos[i]))
            
            # Si el signo es distinto del que debería
            if valor != label[i]:
                # actualizamos w
                w = w + label[i] * datos[i]
                # y decimos que ha habido mejora
                hay_mejora = True
            
        # aumentamos el numero de iteraciones
        itera += 1
    
    return w, itera

#CODIGO DEL ESTUDIANTE

# BLOQUEE DE CÓDIGO PARA ETIQUETAR LOS PUNTOS (COPIA Y PEGA DE OTROS APARTADOS)
etiquetas = []
posibles_etiquetas = (1,-1)
colores = {1:'orange',-1: 'black'}

#etiquetamos cada punto según la función de la recta
for punto in x:
    etiquetas.append(f(punto[0],punto[1],a_ap3,b_ap3))
    
for etiqueta in posibles_etiquetas:
    indice = np.where(np.array(etiquetas) == etiqueta)
    
    plt.scatter(x_ap3[indice,0], x_ap3[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))

###############################################################################

# Le añadimos a x un 0 al principio de cada fila
x_ap3_copia = np.c_[np.ones((x_ap3.shape[0],1), dtype=np.float64),x_ap3]

# w inicial es un vector de 3 0s
w_0 = np.zeros(3)

# Ejecutamos el algoritmo perceptron
w,iteraciones = ajusta_PLA(x_ap3_copia, etiquetas, np.Inf, w_0)

# BLOQUE DE CÓDIGO PARA PINTAR LOS RESULTADOS
plt.plot([-50,50], [a_ap3*-50 + b_ap3, a_ap3*50 + b_ap3], 'k-', label='Recta con la que hemos etiquetado')

plt.plot([-50,50],[ (-w[0]-w[1]*-50)/w[2], (-w[0]-w[1]*50)/w[2]],'y-',label='Recta obtenida con PLA')

plt.title("Nube de 100 puntos bidimensionales en el intervalo {-50} {50}, etiquetados según una recta")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show();
##############################################################################

print ("w: " + str(w) + "\n Iteraciones necesitadas: "+ str(iteraciones))

input("\n--- Pulsar tecla para continuar ---\n")

# Random initializations
iterations = []

# En este bucle for  vamos a calcular un w inicial aleatorio y vamos a lanzar
# el algoritmo perceptron y pintamos la w obtenida y las iteraciones necesarias
for i in range(0,10):
    w_0 = simula_unif(3,1,[0,1]).reshape(1,-1)[0]
    
    print ("Valor inicial: " + str(w_0))
    w,iteraciones = ajusta_PLA(x_ap3_copia,etiquetas,np.Inf,w_0)
    print("\nW obtenida: " + str(w) + " en " + str(iteraciones) + " iteraciones")
    iterations.append(iteraciones)
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


# ESTE BLOQUE DE CÓDIGO HACE LO MISMO QUEE EL DE ARRIBA (ES UN COPIA Y PEGA)
# PERO CON OTRAS ETIQUETAS
# dibujamos los puntos etiquetados, cada uno de un color
for etiqueta in posibles_etiquetas:
    indice = np.where(np.array(etiquetas_ap3b) == etiqueta)
        
    plt.scatter(x_ap3[indice,0], x_ap3[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))
        
w_0 = np.zeros(3)

w,iteraciones = ajusta_PLA(x_ap3_copia, etiquetas_ap3b, 10000, w_0)

plt.plot([-50,50], [a_ap3*-50 + b_ap3, a_ap3*50 + b_ap3], 'k-', label='Recta con la que hemos etiquetado')

plt.plot([-50,50],[ (-w[0]-w[1]*-50)/w[2], (-w[0]-w[1]*50)/w[2]],'r-',label='Recta obtenida con PLA')


plt.title("Nube de 100 puntos bidimensionales en el intervalo {-50} {50}, etiquetados según una recta con ruido")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()

print ("w: " + str(w) + "\n Iteraciones necesitadas: "+ str(iteraciones))

input("\n--- Pulsar tecla para continuar ---\n")

iterations = []

for i in range(0,10):
    w_0 = simula_unif(3,1,[0,1]).reshape(1,-1)[0]
    print ("Valor inicial: " + str(w_0))
    w,iteraciones = ajusta_PLA(x_ap3_copia,etiquetas_ap3b,10000,w_0)
    print("\nW obtenida: " + str(w) + " en " + str(iteraciones) + " iteraciones")
    iterations.append(iteraciones)
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

##################################################################################################

input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# Función que calcula el error empírico
def ERM (x,y,w):
    
    # Sacamos el número de elementos (N)
    num_elementos = x.shape[1]
    
    # inicializamos el error a 0
    error = np.float(0.0)
    
    # Para cada elemento
    for i in range(num_elementos):
        #Calculamos el error siguiendo la fórmula explicada en la memoria
        error += np.log(1 + np.e**(-y[i]*w.T.dot(x[i])))
        
    # Esta operación es parte de la fórmula
    error = error/num_elementos
    
    return error

# Función que calcula el gradiente según la fórmula explicada en la memoria
def grad(x,y,w):
    return (-(y *x)/(1 + np.exp(y * w.T.dot(x))))

# Regresión logística con SGD
def sgdRL(x,y,learning_rate,error2get = 0.01):
    #CODIGO DEL ESTUDIANTE
    # w tendrá tantos 0 como columnas tenga x
    w = np.zeros((x.shape[1],),np.float64)
    
    # Variable que almacenará el w de la iteración anterior
    w_ant = w.copy()
    
    # Variables de control
    acabar = False
    epocas = 0
    
    while not acabar:
        
        # Permutación aleatoria sobre los índices de los datos
        indices_minibatch = np.random.choice(x.shape[0],x.shape[0],replace=False)
        
        # el tamaño de un minibatch es de 1
        # Para cada índice del minibatch aplicamos sgd pero con el gradiente de RL
        for i in indices_minibatch:
            w = w - learning_rate * grad(x[i],y[i],w)
            
        epocas +=1
        
        # Calculamos la distancia para ver si parar o no
        dist = np.linalg.norm(w_ant - w)
        
        # si la distancia es menor que el error a conseguir acabamos
        if dist < error2get:
            acabar = True
            
        w_ant = w.copy()
        
    return w, epocas



#CODIGO DEL ESTUDIANTE

# BLOQUE DE CODIGO COPIA Y PEGA EN EL QUE SACAMOS UNOS DATOS CALCULAMOS UNA RECTA
# ALEATORIA, ETIQUETAMOS LOS PUNTOS Y MOSTRAMOS LSO DATOS
intervalo_trabajo = [0,2]

x = simula_unif(100,2,intervalo_trabajo)

a,b = simula_recta(intervalo_trabajo)

etiquetas = []

posibles_etiquetas = (1,-1)
colores = {1: 'purple',-1: 'pink'}

for punto in x:
    etiquetas.append(f(punto[0],punto[1],a,b))
    
for etiqueta in posibles_etiquetas:
    indice = np.where(np.array(etiquetas) == etiqueta)
    
    plt.scatter(x[indice,0] , x[indice, 1], c=colores[etiqueta], label="{}".format(etiqueta))
    
    
plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] +b], 'b-', label='Recta obteenida aleatoriamente')

plt.title("Nube de 100 puntos bidimensionales en el intervalo [0,2], etiquetados según una recta")
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()

############################################################################################

#Ahora voy a hacer el experimento 100 veces y calcular Eout medio y épocas medias como se pide en el ejericio
input("\n--- Pulsar tecla para continuar ---\n")

contador = 0
max_iter = 100

eout = []
ein = []
epocas = []

# Saco 100 puntos los etiqueto, lanzo RL, calculo Ein, Saco 1000 puntos y con ellos calculo Eout
# Este proceso lo repito 100 veces
while contador < max_iter:
    
    # Genero los datos de training y ejecuto el algoritmo de regresión logística con la recta generada antes
    intervalo_trabajo = [0,2]

    x = simula_unif(100,2,intervalo_trabajo)

    # Bloque de código para etiquetar los puntos
    y = []

    posibles_etiquetas = (1,-1)
    colores = {1: 'purple',-1: 'pink'}

    for punto in x:
        y.append(f(punto[0],punto[1],a,b))
    ###################################################

        
    y = np.array(y)
    
    #Añado un 0 al principio de  cada punto 
    x = np.c_[np.ones((x.shape[0] ,1), dtype=np.float64), x]

    # Lanzo regresión logística
    w,epoc = sgdRL(x,y,0.01,0.01)
    
    # Calculo Ein de esta iteración
    ein.append(ERM(x,y,w))
    
    # Almaceno las épocas necesitadas para esta iteración
    epocas.append(epoc)
    
    # Genero los datos de test y calculo el error
    x_test = simula_unif(10000,2, intervalo_trabajo)

    # Bloque de código para etiquetar
    y_test = []

    posibles_etiquetas = (1,-1)
    colores = {1: 'purple', -1: 'pink'}

    for punto in x_test:
        y_test.append(f(punto[0],punto[1],a,b))
        
    y_test = np.array(y_test)
    
    #####################################################
    
    # Añado un 0 al principio de cada punto
    x_test = np.c_[np.ones((x_test.shape[0],1), dtype=np.float64), x_test]
    
    # Almaceno Eout de esta iteración
    eout.append(ERM(x_test,y_test,w))
    
    contador += 1
    
# Caluclo la media de Ein, Eout y las épocas
print("Ein medio: ", str(np.mean(ein)))
print("Eout medio: ", str(np.mean(eout)))
print("Media de epocas necesitadas para converger: ", str(np.mean(epocas)))


input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

# Función que calcula el porcentaje de puntos mal clasificados
def num_errores_w(x,y,w):
    
    errores = 0
    
    for i in range(len(x)):
        if signo(w.T.dot(x[i])) != y[i]:
            errores += 1
            
    errores = errores/len(x)
    
    return errores

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

# CÓDIGO SACADO DE MI PRÁCTICA 1
# Funcion para calcular el error
def Err(x,y,w):
    
    # He calculado el error según la fórmula dada en teoría (diapositiva 6)
    # Ein(w) = 1/N + SUM(wT*x - y)²
    #
    err = np.square(x.dot(w.T) - y)
    
    return err.mean()

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
###############################################################################

#CODIGO DEL ESTUDIANTE
intervalo_trabajo = [0,1]

# LANZO LA PESUDO INVERS
w  = pseudoinverse (x,y)


#mostramos los datos para training
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]],'y-',label='Recta obtenida con pseudoinversa')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
ax.set_ylim((-7.5,0))
plt.legend()
plt.show()

print ("Error datos de training usando pseudoinversa: ", Err(x,y,w))


input("\n--- Pulsar tecla para continuar ---\n")
# mostramos los datos para test
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con pseudoinversa')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.xlim(intervalo_trabajo)
plt.ylim([-7.5, 0])
plt.legend()
plt.show()

print ("Error datos de test usando pseudoinversa: ", Err(x_test,y_test,w))

input("\n--- Pulsar tecla para continuar ---\n")

#POCKET ALGORITHM
def pocket(x,y,iteraciones,w_ini):
    
    #establecemos el w inicial
    w_mejor = w_ini.copy()
    #calculamos el mejor errro hasta ahora
    ein_w_mejor = num_errores_w(x,y,w_mejor)
    w = w_mejor.copy()
    
    it = 0
    
    #mientras queden iteraciones
    while it < iteraciones:
        #ajustamos con PLA 1 única iteración (explicación en la memoria)
        w,nada = ajusta_PLA(x,y,1,w.copy())
        ein_w = num_errores_w(x,y,w)
        
        #si ha mejorado actualizamos
        if ein_w < ein_w_mejor:
            w_mejor = w.copy()
            ein_w_mejor = ein_w
            
        it += 1
        
    return w_mejor
      
#CODIGO DEL ESTUDIANTE
w = pocket(x,y,200,w.copy())

#mostramos los datos para training
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]],'y-',label='Recta obtenida con POCKET')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
ax.set_ylim((-7.5,0))
plt.legend()
plt.show()

ein = num_errores_w(x,y,w)
print("Error de pocket dentro de la muestra: ", ein)


input("\n--- Pulsar tecla para continuar ---\n")

# mostramos los datos para test
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con pocket')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.xlim(intervalo_trabajo)
plt.ylim([-7.5, 0])
plt.legend()
plt.show()

etest = num_errores_w(x_test,y_test,w)
print("Error de pocket fuera de la muestra: ", etest)

input("\n--- Pulsar tecla para continuar ---\n")

#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE

# calculamos las cotas con el delta del enunciado y utilizando las fórmulas vistas en teoría
delta = 0.05

cota_ein = ein + np.sqrt((1/(2*len(x))) * np.log(2/delta))

print("Cota de Eout usando Ein: ", cota_ein)

input("\n--- Pulsar tecla para continuar ---\n")

cota_etest = etest + np.sqrt((1/(2*len(x))) * np.log(2/delta))

print("Cota de Eout usando Etest: ", cota_etest)

