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


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE
plt.scatter(x[:,0],x[:, 1], c='orange')
plt.title("Nube de 50 puntos simula_unif dimensión 2 rango -50,50")
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()
plt.clf()

x = simula_gaus(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE
plt.scatter(x[:,0],x[:,1],c='orange')
plt.title("Nube de 50 puntos simula_gaus dimensión 2 sigma [5,7]")
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

plt.clf()

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
x = simula_unif(100,2,[-50,50])

x_ap3 = x.copy();

a,b = simula_recta([-50,50])

a_ap3 = a.copy()
b_ap3 = b.copy()

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

# dibujamos los puntos etiquetados, cada uno de un color
for etiqueta in posibles_etiquetas:
        indice = np.where(np.array(etiquetas) == etiqueta)
        
        plt.scatter(x[indice,0], x[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))
        
        
plt.plot([-50,50],[a*-50 +b, a*50 +b], 'k-', label='Recta aleatoria')

plt.title("Nube de 100 puntos bidimensionales en el intervalo {-50} {50}, etiquetados según una recta")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE
def ruido(etiquetas):
    indices_pos = np.where(np.array(etiquetas) == 1)
    indices_pos = indices_pos[0]
    
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

etiquetas = ruido(etiquetas)
etiquetas_ap3b = etiquetas.copy()
    
# dibujamos los puntos etiquetados, cada uno de un color
for etiqueta in posibles_etiquetas:
        indice = np.where(np.array(etiquetas) == etiqueta)
        
        plt.scatter(x[indice,0], x[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))
        
        
plt.plot([-50,50],[a*-50 +b, a*50 +b], 'k-', label='Recta aleatoria')

plt.title("Nube de 100 puntos bidimensionales en el intervalo {-50} {50}, etiquetados según una recta con ruido")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()

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
    
    
def f1(x):
    return np.float64((x[:, 0]-10)**2 + (x[:, 1] - 20)**2 -400)
    
def f2(x):
    return np.float64(0.5*(x[:, 0] + 10)**2 + (x[:, 1] - 20)**2 -400)

def f3(x):
    return np.float64(0.5*(x[:,0] - 10)**2 - (x[:,1] + 20)**2 -400)

def f4(x):
    return np.float64(x[:,1] - (20*(x[:,0]**2)) - (5*x[:,0]) + 3)
    
#CODIGO DEL ESTUDIANTE
etiquetas_3 = etiquetas.copy()

plot_datos_cuad(x,etiquetas_3,f1,"Funcion f1 con etiquetas de la recta")


input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,etiquetas_3,f2,"Funcion f2 con etiquetas de la recta")


input("\n--- Pulsar tecla para continuar ---\n")


plot_datos_cuad(x,etiquetas_3,f3,"Funcion f3 con etiquetas de la recta")

input("\n--- Pulsar tecla para continuar ---\n")

plot_datos_cuad(x,etiquetas_3,f4,"Funcion f4 con etiquetas de la recta")

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    w = vini.copy()
    itera = 0;
    hay_mejora = True
    
    
    while hay_mejora and itera < max_iter:
        hay_mejora  = False
        
        for i in range(0, len(datos)):
            valor = signo(w.T.dot(datos[i]))
            
            if valor != label[i]:
                w = w + label[i] * datos[i]
                hay_mejora = True
            
        itera += 1
    
    return w, itera

#CODIGO DEL ESTUDIANTE
#x = simula_unif(100,2,[-50,50])

#a,b = simula_recta([-50,50])

etiquetas = []
posibles_etiquetas = (1,-1)
colores = {1:'orange',-1: 'black'}

#etiquetamos cada punto según la función de la recta
for punto in x:
    etiquetas.append(f(punto[0],punto[1],a_ap3,b_ap3))
    
for etiqueta in posibles_etiquetas:
    indice = np.where(np.array(etiquetas) == etiqueta)
    
    plt.scatter(x_ap3[indice,0], x_ap3[indice,1],c=colores[etiqueta],label="{}".format(etiqueta))

x_ap3_copia = np.c_[np.ones((x_ap3.shape[0],1), dtype=np.float64),x_ap3]

w_0 = np.zeros(3)

w,iteraciones = ajusta_PLA(x_ap3_copia, etiquetas, np.Inf, w_0)

plt.plot([-50,50], [a_ap3*-50 + b_ap3, a_ap3*50 + b_ap3], 'k-', label='Recta con la que hemos etiquetado')

plt.plot([-50,50],[ (-w[0]-w[1]*-50)/w[2], (-w[0]-w[1]*50)/w[2]],'y-',label='Recta obtenida con PLA')

plt.title("Nube de 100 puntos bidimensionales en el intervalo {-50} {50}, etiquetados según una recta")
plt.legend()
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show();

print ("w: " + str(w) + "\n Iteraciones necesitadas: "+ str(iteraciones))

input("\n--- Pulsar tecla para continuar ---\n")

# Random initializations
iterations = []
for i in range(0,10):
    w_0 = simula_unif(3,1,[0,1]).reshape(1,-1)[0]
    w,iteraciones = ajusta_PLA(x_ap3_copia,etiquetas,np.Inf,w_0)
    iterations.append(iteraciones)
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE

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
    w,iteraciones = ajusta_PLA(x_ap3_copia,etiquetas_ap3b,10000,w_0)
    print(iteraciones)
    iterations.append(iteraciones)
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
def ERM (x,y,w):
    
    num_elementos = x.shape[1]
    
    error = np.float(0.0)
    
    for i in range(num_elementos):
        error += np.log(1 + np.e**(-y[i]*w.T.dot(x[i])))
        
    
    error = error/num_elementos
    
    return error

def grad(x,y,w):
    return -(y *x)/(1 + np.exp(y * w.T.dot(x)))

def sgdRL(x,y,learning_rate,error2get = 0.01):
    #CODIGO DEL ESTUDIANTE
    w = np.zeros((x.shape[1],),np.float64)
    
    w_ant = w.copy()
    
    acabar = False
    epocas = 0
    
    while not acabar:
        
        indices_minibatch = np.random.choice(x.shape[0],x.shape[0],replace=False)
        
        for i in indices_minibatch:
            w = w - learning_rate *grad(x[i],y[i],w)
            
        epocas +=1
        
        dist = np.linalg.norm(w_ant - w)
        
        if dist < error2get:
            acabar = True
            
        w_ant = w.copy()
        
    return w, epocas



#CODIGO DEL ESTUDIANTE
intervalo_trabajo = [0,2]

x = simula_unif(100,2,intervalo_trabajo)

puntos_recta = np.random.choice(x.shape[0],2,replace=False)

a = (x[puntos_recta[1]][1] - x[puntos_recta[0]][1]) / (x[puntos_recta[1]][0]- x[puntos_recta[0]][0])

b = x[puntos_recta[0]][1] - (a*x[puntos_recta[0]][0])

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

input("\n--- Pulsar tecla para continuar ---\n")

etiquetas = np.array(etiquetas)

x = np.c_[np.ones((x.shape[0] ,1), dtype=np.float64), x]

w,epocas = sgdRL(x,etiquetas,0.01,0.01)

posibles_etiquetas = (1,-1)
colores = {1 :'purple',-1: 'pink'}

plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]],'b-',label='Recta obtenida con sgdRL')

plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1]+b],"r-", label='Recta obtenida aleatoriamente')

for etiqueta in posibles_etiquetas:
    indice = np.where(np.array(etiquetas) == etiqueta)
    
    plt.scatter(x[indice,1],x[indice,2], c=colores[etiqueta], label="{}".format(etiqueta))
    
plt.title("Nube dee 100 puntos bidimensionales en el intervalo [0,2], etiquetados segun una recta")
plt.legend(loc=4)
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()


print("w: " + str(w) + "\nÉpocas: " + str(iteraciones))
print("Error obtenido dentro de la muestra (Ein): " + str(ERM(x, etiquetas, w)))

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

input("\n--- Pulsar tecla para continuar ---\n")

#CODIGO DEL ESTUDIANTE
x_test = simula_unif(1000,2, intervalo_trabajo)

etiquetas = []

posibles_etiquetas = (1,-1)
colores = {1: 'purple', -1: 'pink'}

for punto in x_test:
    etiquetas.append(f(punto[0],punto[1],a,b))
    
etiqueetas = np.array(etiquetas)
x_test = np.c_[np.ones((x_test.shape[0],1), dtype=np.float64), x_test]

plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]],'b-',label='Recta obtenida con sgdRL')

plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1]+b],"r-", label='Recta obtenida aleatoriamente')

for etiqueta in posibles_etiquetas:
    indice = np.where(np.array(etiquetas) == etiqueta)
    
    plt.scatter(x_test[indice,1],x_test[indice,2], c=colores[etiqueta], label="{}".format(etiqueta))
    
plt.title("Nube dee 100 puntos bidimensionales en el intervalo [0,2], etiquetados segun una recta")
plt.legend(loc=4)
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor de x de los puntos obtenidos")
plt.ylabel("Valor de y de los puntos obtenidos")

plt.show()

print("Error fuera de la muestra (Eout): " +str(ERM(x_test,etiquetas,w)))
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


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

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
