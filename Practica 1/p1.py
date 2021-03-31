# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Pilar Navarro Ramírez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(1)

def continuar():
    input("\n----Presiona Enter para continuar----\n")

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 2\n')
#Ejercicio 1 y 2

#El parámetro w será un array o tupla con las coordenadas (u,v) donde se evalúa la función

def E(w):
    u,v=w[0],w[1]
    return (u**3*np.exp(v-2)-2*v**2*np.exp(-u))**2  

#Derivada parcial de E con respecto a u
def dEu(w):
    u,v=w[0],w[1]
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(3*u**2*np.exp(v-2)+2*v**2*np.exp(-u))
    
#Derivada parcial de E con respecto a v
def dEv(w):
    u,v=w[0],w[1]
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(u**3*np.exp(v-2)-4*v*np.exp(-u))

#Gradiente de E
def gradE(w):
    u,v=w[0],w[1]
    return np.array([dEu(w), dEv(w)],dtype='float64')

def gradient_descent(init,f,grad,error2get,maxIter,lr):
    """ Función que implementa el algoritmo de gradiente descendente
        
        Args:
            init: punto inicial del que parte el algoritmo
            f: función a la que se aplica el algoritmo
            grad: gradiente de la función f
            error2get: margen de error 
                (mínimo valor de la función f que se desea alcanzar)
            maxIter: máximo número de iteraciones del algoritmo
            lr: learning rate (tasa de aprendizaje)
        
        Returns:
            w: primer punto donde se alcanza un valor de la función f
             igual o menor a 'error2get'. 
             En el caso en que este valor no se alcance en el número
             de iteraciones indicado por maxIter, w es el punto de 
             la última iteración
            
            iterations: número de iteraciones necesarias para alcanzar
             un valor de la función f menor o igual a 'error2get'. 
             En el caso en que este valor no se alcance en un número 
             de iteraciones de maxIter, iterations coindice con maxIter.
    
    """
   #Inicializamos las coordenadas con el valor que se pasa como parámetro
    w=init
    
    #Iteramos hasta alcanzar maxIter
    for i in range(1,maxIter+1):
        #Actualizamos las coordenadas en la dirección del gradiente descendente
        w=w-lr*grad(w)
        
        #Paramos si el valor de la función es menor o igual que error2get
        #En ese caso se devuelve el número de iteraciones hasta ese momento
        # y las coordenadas del punto obtenido
        if f(w)<=error2get: 
            return w, i
    
    #Si se alcanza el máximo número de iteraciones y no se ha encontrado
    # un punto donde la función tome un valor menor o igual que el deseado,
    # se devuelve el punto de la última iteración y maxIter
    return w, maxIter    


#Learning Rate
eta = 0.1 
#Máximo número de iteraciones
maxIter = 100
# Margen de error
error2get = 1e-14
#Punto incial para aplicar el algoritmo de gradiente descendente
initial_point = np.array([1.0,1.0],dtype='float64')

#Aplicamos el algoritmo de gradiente descendente a la función E(u,v) 
w, it = gradient_descent(initial_point,E,gradE,error2get,maxIter,eta)
print("---Apartado b)---")
#b) Iteraciones necesarias para obterner un valor de la función E(u,v) menor o igual a 1e-14
print ('Numero de iteraciones: ', it)
print("---Apartado c)---")
#c) Coordenadas donde se alcanzó por primera vez un valor de la función menor o igual a 1e-14
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print('Valor de la función E(u,v) en ese punto: ', E(w))

continuar()
# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E([X,Y]) #E_w([X, Y])
fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E([min_point_[0], min_point_[1]]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

continuar()

print('Ejercicio 3\n')

#Función f(x,y)
def f(w):
    x,y=w[0],w[1]
    return (x+2)**2+2*(y-2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
#Derivada parcial de f con respecto a x
def fx(w):
    x,y=w[0],w[1]
    return 2*(x+2)+4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
#Derivada parcial de f con respecto a y
def fy(w):
    x,y=w[0],w[1]
    return 4*(y-2)+4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
#Gradiente de la función f
def gradf(w):
    return np.array([fx(w),fy(w)],dtype='float64')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-10,10, 50)
y = np.linspace(-10,10, 50)
X, Y = np.meshgrid(x, y)
Z = f([X,Y]) #E_w([X, Y])
fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], f([min_point_[0], min_point_[1]]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()

continuar()

def plot_gd(init,f,grad,maxIter,lr):
    
    """ Función que implementa el algoritmo de gradiente descendente y genera
        un gráfico con la evolución del valor de la función f en cada
        iteración del algoritmo.
        
        Args:
            init: punto inicial del que parte el algoritmo
            f: función a la que se aplica el algoritmo
            grad: gradiente de la función f
            maxIter: máximo número de iteraciones del algoritmo
            lr: learning rate (tasa de aprendizaje)
    
    """
    #imagenes es un vector con los valores de la función en cada punto
    #le añadimos en primer lugar el valor de la función en el punto inicial
    imagenes=np.array(f(init),np.float)
    #Inicializamos las coordenadas con el valor que se pasa como parámetro
    w=init
        
    #Iteramos hasta alcanzar maxIter
    for i in range(maxIter):
        #Actualizamos las coordenadas en la dirección del gradiente descendente
        w=w-lr*grad(w)
        #Añadimos al vector de valores el valor de la función en las nuevas coordenadas
        imagenes=np.append(imagenes,f(w))
    
    #Generamos la gráfica
    plt.figure(figsize=(7,7))
    plt.plot(range(maxIter+1),imagenes,color='mediumspringgreen',marker='o')
    plt.title(f'Gráfica con Learning Rate de {lr}')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()

#Punto incial para aplicar el algoritmo de gradiente descendente
initial_point = np.array([-1.0,1.0],dtype='float64')
maxIter = 50 #Máximo número de iteraciones

print("----Apartado a)-----")
print("Gráfica con learning rate de 0.01")
eta1 = 0.01 #Learning Rate
plot_gd(initial_point,f,gradf,maxIter,eta1)

continuar()

print("Gráfica con learning rate de 0.1")
eta2 = 0.1 #Learning Rate
plot_gd(initial_point,f,gradf,maxIter,eta2)
    
continuar()

def gd(init,f,grad,maxIter,lr):
    """ Función que implementa el algoritmo de gradiente descendente
        
        Args:
            init: punto inicial del que parte el algoritmo
            f: función a la que se aplica el algoritmo
            grad: gradiente de la función f
            maxIter: máximo número de iteraciones del algoritmo
            lr: learning rate (tasa de aprendizaje)
        
        Returns:
            w: punto obtenido tras aplicar el algoritmo de gradiente 
            descendente un número de maxIter iteraciones
    """
    #Inicializamos las coordenadas con el valor que se pasa como parámetro
    w=init
    #Iteramos hasta alcanzar maxIter
    for i in range(maxIter):
        #Actualizamos las coordenadas en la dirección del gradiente descendente
        w=w-lr*grad(w)
    #Devolvemos las coordenadas obtenidas tras maxIter iteraciones del algoritmo
    return w

eta = 0.01 #Learning Rate
maxIter = 50 #Máximo número de iteraciones 

print("---Apartado b)---")
print("Punto de inicio = (-0.5,-0.5)")
initial_point1 = np.array([-0.5,-0.5],dtype='float64')
w=gd(initial_point1,f,gradf,maxIter,eta)
print("Valor mínimo: ",f(w))
print("Coordenadas donde se alcanza el valor mínimo:\n (", w[0],',',w[1],')')

continuar()
print("Punto de inicio = (1,1)")
initial_point2 = np.array([1.0,1.0],dtype='float64')
w=gd(initial_point2,f,gradf,maxIter,eta)
print("Valor mínimo: ",f(w))
print("Coordenadas donde se alcanza el valor mínimo:\n (", w[0],',',w[1],')')

continuar()

print("Punto de inicio = (2.1,-2.1)")
initial_point3 = np.array([2.1,-2.1],dtype='float64')
w=gd(initial_point3,f,gradf,maxIter,eta)
print("Valor mínimo: ",f(w))
print("Coordenadas donde se alcanza el valor mínimo:\n (", w[0],',',w[1],')')

continuar()

print("Punto de inicio = (-3,3)")
initial_point4 = np.array([-3.0,3.0],dtype='float64')
w=gd(initial_point4,f,gradf,maxIter,eta)
print("Valor mínimo: ",f(w))
print("Coordenadas donde se alcanza el valor mínimo:\n (", w[0],',',w[1],')')

continuar()

print("Punto de inicio = (-2,2)")
initial_point5 = np.array([-2.0,2.0],dtype='float64')
w=gd(initial_point5,f,gradf,maxIter,eta)
print("Valor mínimo: ",f(w))
print("Coordenadas donde se alcanza el valor mínimo:\n (", w[0],',',w[1],')')

continuar()

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

# Funcion para calcular el error de un modelo de regresión lineal
def Err(x,y,w):
    #np.linalg.norm calcula la norma (por defecto, norma euclídea) de un vector o matriz
    #np.dot calcula el producto escalar
     return (np.linalg.norm(np.dot(x,w)-y)**2)/len(x)

# Gradiente Descendente Estocástico
def sgd(x,y,lr,maxIter,minibatch_size):
    """ Función que implementa el algoritmo de gradiente descendente estocástico
     en un modelo de regresión lineal

    Args:
        x: matriz con las características de cada dato
        y: etiquetas asociadas a cada dato
        maxIter: máximo número de iteraciones del algoritmo
        lr: learning rate (tasa de aprendizaje)
        minibatch_size: tamaño de un mini-batch

    Returns:
        w: vector de pesos obtenido por el algoritmo
    """
    #Inicializamos los pesos como un vector de ceros
    #con una longitud igual a la de cada vector de caracaterísticas x_i
    w = np.zeros(x.shape[1],dtype='float64')
    # Consideramos un vector de índices permutados aleatoriamente,
    # con tantos índices como elementos hay en x (en nuestro dataset)
    indices=np.random.permutation(x.shape[0]) 
    # Posición del vector de índices a partir de la cual comienza un mini-batch
    batch_index=0
    for i in range(maxIter):
        # Posición en el vector de índices del último elemento del mini-batch actual
        batch_end=batch_index+minibatch_size
        # Nos quedamos con los elementos de x e y que se encuentran en las posiciones
        # que indican los correspondientes índices del vector de índices permutados
        x_batch,y_batch = x[indices[batch_index:batch_end]],y[indices[batch_index:batch_end]]
        #Calculamos el gradiente del error del modelo de regresión lineal
        grad_err=2*np.dot(x_batch.T,np.dot(x_batch,w)-y_batch)/len(x_batch)
        #Actualizamos los pesos en la dirección del gradiente descendente del error
        w=w-lr*grad_err
        #Actualizamos el índice del inicio del siguiente mini-batch
        batch_index=batch_end
        #Si hemos recorrido el dataset completo (superamos el tamaño del vector de índices),
        # se vuelven a permutar los índices
        if batch_index>=len(indices):
            indices=np.random.permutation(x.shape[0]) 
            batch_index=0

    return w

# Algoritmo de la Pseudo-inversa
def pseudoinverse(x,y):
    #np.linalg.pinv calcula la matriz pseudo-inversa de la matriz x
    #np.dot computa el producto de la matriz pseudo-inversa de x por y
    return np.dot(np.linalg.pinv(x),y)

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

# Para implementar esta función me he basado en el siguiente tutorial:
# https://likegeeks.com/3d-plotting-in-python/
def plot_plano3d(w,x,y,title,axis_labels,legend_labels):
    """ Función que genera un gráfico en 3d con los datos de entrenamiento
        usados en el ajuste lineal junto con el plano correspondiente a 
        la función lineal determinada por los pesos w

    Args:
        x: matriz con las características de cada dato de entrenamiento
        y: etiquetas asociadas a cada dato de entrenamiento
        w: vector de pesos que determina a una función lineal
        title: título para el gráfico generado
        axis_labels: lista de etiquetas para los ejes
        legend_labels: lista de etiquetas para la leyenda
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d') # Creamos los ejes 3D
    
    ax.scatter(x[:,1],x[:,2],y,color='violet') # Visualizamos los datos  
    
    #Ponemos etiquetas a los ejes
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    
    #Determinamos los extremos de los valores de las características
    min_int = np.min(x[:,1])
    max_int = np.max(x[:,1])
    min_sim = np.min(x[:,2])
    max_sim = np.max(x[:,2])
    
    #Creamos una malla de puntos equiespaciados donde evaluar la función
    xx, yy = np.meshgrid(np.linspace(min_int,max_int,20),np.linspace(min_sim,max_sim,20))
    
    #Evaluamos en los puntos de la malla la función lineal obtenida 
    z = np.array(w[0]+xx*w[1]+yy*w[2])
    
    # Visualizamos el hiperplano 
    ax.plot_surface(xx,yy, z, color='aqua',alpha=0.6)
    
    # Creamos un gráfico en 2D que no muestre nada para poder poner leyenda,
    # ya que la leyenda no admite el tipo devuelto por un scatter 3D, según se comenta en el siguiente enlace
    # https://stackoverflow.com/questions/20505105/add-a-legend-in-a-3d-scatterplot-with-scatter-in-matplotlib
    scatter_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='violet', marker = 'o')
    plane_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='aqua', marker = '_')
    ax.legend([scatter_proxy, plane_proxy], legend_labels, numpoints = 1, loc='upper left')
    
    #Cambiamos la posición del gráfico para que se vean mejor los datos y el hiperplano
    #https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
    ax.view_init(11,6) 
    
    plt.title(title)
    plt.show()
    
def plot_plano2d(x,y,title,lim,axis_labels,legend_labels,*w):
    """ Función que genera un gráfico en 2d con los datos de entrenamiento
        usados en el ajuste lineal junto con las funciones lineales
        determinadas por los vectores de pesos w

    Args:
        x: matriz con las características de cada dato de entrenamiento
        y: etiquetas asociadas a cada dato de entrenamiento
        title: título para el gráfico generado
        lim: booleano que determina si se establecen límites a los valores de los ejes
        axis_labels: lista de etiquetas para los ejes
        legend_labels: lista de etiquetas para la leyenda
        w: vectores de pesos que determinan diferentes funciones lineales
        
        
    """
    plt.figure(figsize=(10,10))
    plt.title(title)
    
    #Ponemos etiquetas a los ejes
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    
    # Creamos dos arrays, cada uno con los datos de entrenamiento de una clase
    clase1 = np.array([a for a,b in zip(x,y) if b == 1])
    clase2 = np.array([a for a,b in zip(x,y) if b == -1])

    # Visualizamos los datos de entrenamiento,
    # poniendo un color diferente a cada clase
    plt.scatter(clase1[:,1], clase1[:,2], c="lightgreen", label=legend_labels[0])
    plt.scatter(clase2[:,1], clase2[:,2], c="purple", label=legend_labels[1])

    # Nos quedamos con los valores de la primera característica (intensidad) del conjunto de datos
    xx=x[:,1]
    # Intersecamos el plano determinado por el vector de pesos con el plano z=0 
    # (para representarlo en 2d), despejamos la segunda característica (simetría), como aparece en
    # https://stackoverflow.com/questions/42704698/logistic-regression-plotting-decision-boundary-from-theta
    # y aplicamos el resultado a la primera característica (intensidad) para obtener la segunda
    yy = [(-w[0][0]-w[0][1]*i)/w[0][2] for i in xx]
    
    if lim: # Si el parámetro lim es True
        #Determinamos los extremos de los valores de las características
        min_x1= np.min(x[:,1])
        max_x1 = np.max(x[:,1])
        min_x2= np.min(x[:,2])
        max_x2 = np.max(x[:,2])

        # Establecemos los límites de los ejes
        plt.xlim(min_x1,max_x1)
        plt.ylim(min_x2,max_x2)
    
    # Visualizamos la solución
    plt.plot(xx, yy, label=legend_labels[2], color='aqua') 
    
    if len(w)>1: # Si nos dan más de un vector de pesos,
        # repetimos el procedimiento para esos valores de los pesos
        yy = [(-w[1][0]-w[1][1]*i)/w[1][2] for i in xx]
        plt.plot(xx, yy, label=legend_labels[3], color='darkblue') 
    
    plt.legend()
    plt.show()

# Vector con distintos tamaños de mini-batch
batches=np.array([1,16,32,64,256,128])
# Vector con las etiquetas de los ejes
axis_labels=['Valor medio del nivel de gris',
                'Simetría respecto al eje vertical',
                'Etiqueta \n (-1 para el dígito 1, 1 para el dígito 5)']
# Vector con las etiquetas que queremos que aparezcan en la leyenda de los gráficos generados
legend_labels=['Datos usados en el ajuste', 'Solución obtenida']
print('Gradiente descendente estocástico')
# Iteramos sobre el vector de tamaños de mini-batch,
# para aplicar el algoritmo de sgd con distintos tamaños y analizar los resultados
for tam in batches:  
    w1 = sgd(x,y,0.01,1000000,tam)
    print(f'Tamaño de batch de {tam}')
    print("Vector de pesos w:\n",w1)

    print ('\nBondad del resultado:\n')
    print ("Ein: ", Err(x,y,w1))
    print ("Eout: ", Err(x_test, y_test, w1))

    continuar()
# Representamos la solución con tamaño de batch de 128,
# que es el último elemento del vector de batches al que se aplica el algoritmo
plot_plano3d(w1,x,y,f'Ajuste con el algoritmo de SGD\n Tamaño batch = 128',axis_labels,legend_labels)

continuar()

w = sgd(x,y,0.01,1000000,len(x))
print('Tamaño de batch igual al de todo el conjunto de datos')
print("Vector de pesos w:\n",w)

print ('\nBondad del resultado:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

plot_plano3d(w,x,y,'Ajuste con el algoritmo de Batch GD',axis_labels,legend_labels)

continuar()

print('Algoritmo de la pseudo-inversa')
w2 = pseudoinverse(x,y)
print("Vector de pesos w óptimo:\n",w2)

print ('\nBondad del resultado:\n')
print ("Ein: ", Err(x,y,w2))
print ("Eout: ", Err(x_test, y_test, w2))

plot_plano3d(w2,x,y,'Ajuste con el algoritmo de la pseudoinversa',axis_labels,legend_labels)

continuar()

axis_labels2d=['Valor medio del nivel de gris',
            'Simetría respecto al eje vertical']


legend_labels2d=['Dígito 5 = Etiqueta 1',
                'Dígito 1 = Etiqueta -1',
                'Stochastic Gradient Descent',
                'Pseudo-inversa']

# Representamos en 2d las soluciones obtenidas con el algoritmo de la pseudoinversa
# y con el algoritmo de sgd para tamaño de batch de 128
plot_plano2d(x,y,'Comparación de las soluciones obtenidas',False,axis_labels2d,legend_labels2d,w1,w2)
continuar()

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))


#EXPERIMENTO
print("EXPERIMENTO\n---Apartado a)---")
#Generamos la muestra
sample=simula_unif(1000,2,1)

#Visualizamos los puntos obtenidos
plt.figure(figsize=(7,7))
plt.scatter(sample[:,0],sample[:,1],c='turquoise')
plt.title("Muestra de 1000 puntos en el cuadrado [-1,1]x[-1,1]")
plt.show()

continuar() 
print("---Apartado b)---")

#Función signo
def sign(x):
	if x >= 0:
		return 1
	return -1

#Función f pedida 
def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6)

# Asignamos etiquetas a la muestra generada
etiquetas=np.array([f(x1,x2) for x1,x2 in sample],dtype='float64')

# Añadimos ruido a las etiquetas
# Creamos un vector aleatorio de índices con un tamaño igual al 10% de las etiquetas
indices=np.random.randint(len(etiquetas),size=int(len(etiquetas)*0.1))
# Cambiamos el signo de las etiquetas cuyas posiciones en el vector de etiquetas
# se encuentren en el vector aleatorio de índices creado
for i in indices:
    etiquetas[i]=-etiquetas[i]


# Creamos dos arrays, cada uno con los datos de la muestra de una clase
positivos = np.array([x for x,y in zip(sample,etiquetas) if y == 1])
negativos = np.array([x for x,y in zip(sample,etiquetas) if y == -1])

# Visualización de las etiquetas

plt.figure(figsize=(7,7))

# Ponemos un color diferente a cada clase
plt.scatter(positivos[:,0], positivos[:,1], c="lightgreen", label="Etiqueta 1")
plt.scatter(negativos[:,0], negativos[:,1], c="purple", label="Etiqueta -1")
plt.title("Muestra de 1000 puntos en el cuadrado [-1,1]x[-1,1]\n con sus repectivas etiquetas")
plt.legend(loc='upper right')
plt.show()

continuar()

print("---Apartado c)---")

# Establecemos el vector de características
x=np.array([[1,x1,x2] for x1,x2 in sample],dtype='float64')

# Ajustamos un modelo de regresión lineal con el algoritmo de sgd
w=sgd(x,etiquetas,0.01,10000,32)

print("Vector de pesos w obtenido con el algoritmo de SGD:\n",w)
print ('\nBondad del resultado:\n')
print ("Ein: ", Err(x,etiquetas,w))

continuar()

# Visualizamos el plano obtenido en el ajuste en 3d y 2d
title='Ajuste con el algoritmo de SGD\n Características lineales'

axis_labels=['Coordenada x',
            'Coordenada y',
            'Etiqueta']

legend_labels=['Datos usados en el ajuste', 'Solución obtenida']

axis_labels2d=['Coordenada x','Coordenada y']
legend_labels2d=['Etiqueta 1','Etiqueta -1','Stochastic Gradient Descent']

plot_plano3d(w,x,etiquetas,title,axis_labels,legend_labels)
continuar()
plot_plano2d(x,etiquetas,title,True,axis_labels2d,legend_labels2d,w)

continuar()

print('---Apartado d)---\n')

def datos():
    """ Función que genera una muestra de 100 puntos
        uniformemente distribuidos en el cuadrado [-1,1]x[-1,1],
        le asigna una etiqueta a los mismos según la función f (ya definida)
        e introduce algo de ruido en dichas etiquetas. 
        
        Returns:
            x: vector de características correspondiente a la muestra generada
            etiquetas: vector de etiquetas (con ruido) asociadas a los puntos de x
    
    """  
    #Generamos la muestra
    sample=simula_unif(1000,2,1)
    # Asignamos etiquetas a la muestra generada
    etiquetas=np.array([f(x1,x2) for x1,x2 in sample],dtype='float64')
    # Añadimos ruido a las etiquetas
    # Creamos un vector aleatorio de índices con un tamaño igual al 10% de las etiquetas
    indices=np.random.randint(len(etiquetas),size=int(len(etiquetas)*0.1))
    # Cambiamos el signo de las etiquetas cuyas posiciones en el vector de etiquetas
    # se encuentren en el vector aleatorio de índices creado
    for i in indices:
        etiquetas[i]=-etiquetas[i]
    # Establecemos el vector de características
    x=np.array([[1,x1,x2] for x1,x2 in sample],dtype='float64')
    
    return x,etiquetas


#Inicializamos a 0 las medias de los errores 
media_Ein=0
media_Eout=0
#Repetimos 1000 veces el experimento
for i in range(1000):
    # Generamos los datos de entrenamiento
    x,y=datos()
    # Estimamos los pesos con el algoritmo de sgd
    w=sgd(x,y,0.01,10000,32)
    # Añadimos el error en la muestra obtenido en esta iteración
    media_Ein+=Err(x,y,w)
    # Generamos los datos de test
    x_test,y_test=datos()
    # Añadimos el error en el conjunto de test obtenido en esta iteración
    media_Eout+=Err(x_test,y_test,w)

# Calculamos las medias de los errores 
media_Ein=media_Ein/1000; media_Eout=media_Eout/1000

print("Errores medios tras 1000 repeticiones del experimento con características lineales")
print("Ein medio:",media_Ein)
print("Eout medio:",media_Eout)

continuar()

print("Experimento con características no lineales\n")

# Generamos los datos de entrenamiento, con el vector de características lineales
xlin,y=datos()
# Consideramos el siguiente vector de características no lineales
x=np.array([[1,x1,x2,x1*x2,x1**2,x2**2] for x1,x2 in xlin[:,1:]],dtype='float64')
# Ajustamos el nuevo modelo de regresión lineal con el algoritmo de SGD
w=sgd(x,y,0.01,10000,32)

print("Vector de pesos w obtenido con el algoritmo de SGD para características no lineales:\n",w)
print ('\nBondad del resultado:\n')
print ("Ein: ", Err(x,y,w))

continuar()


# Representamos ahora en 3d la solución obtenida junto con los datos usados para el ajuste
print('Gráfico en 3D de la solución obtenida\ntras el ajuste de un modelo de regresión lineal con características no lineales')
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d') # Creamos los ejes 3D

# Visualizamos los datos  
ax.scatter(x[:,1],x[:,2],y,color='violet') 

#Ponemos etiquetas a los ejes
ax.set_xlabel('Coordenada x')
ax.set_ylabel('Coordenada y')
ax.set_zlabel('Etiqueta')

#Creamos una malla de puntos equiespaciados donde evaluar la función 
xx, yy = np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))

#Evaluamos la función obtenida en los puntos de la malla
z = np.array(w[0]+xx*w[1]+yy*w[2]+xx*yy*w[3]+xx**2*w[4]+yy**2*w[5])

# Visualizamos la solución
ax.plot_surface(xx,yy, z, color='aqua',alpha=0.6)

# Creamos un gráfico en 2D que no muestre nada para poder poner leyenda,
# ya que la leyenda no admite el tipo devuelto por un scatter 3D, según se comenta en el siguiente enlace
# https://stackoverflow.com/questions/20505105/add-a-legend-in-a-3d-scatterplot-with-scatter-in-matplotlib
scatter_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='violet', marker = 'o')
plane_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='aqua', marker = '_')
ax.legend([scatter_proxy, plane_proxy], ['Datos usados en el ajuste','Solución obtenida'], numpoints = 1, loc='upper left')

#Cambiamos la posición del gráfico para que se vean mejor los datos y la solución
#https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
ax.view_init(17,14) 

plt.title('Ajuste con el algoritmo de SGD\n Características no lineales')
plt.show()
    
continuar()

# Representamos ahora lo mismo pero en 2d, es decir, proyectando en el plano z=0
print('Gráfico en 2D de la solución obtenida\ntras el ajuste de un modelo de regresión lineal con características no lineales')
plt.figure(figsize=(8,8))

#Ponemos etiquetas a los ejes
plt.xlabel('Coordenada x')
plt.ylabel('Coordenada y')

plt.title('Ajuste con el algoritmo de SGD\n Características no lineales')

# Creamos dos arrays, cada uno con los datos de entrenamiento de una clase
positivos = np.array([a for a,b in zip(x,y) if b == 1])
negativos = np.array([a for a,b in zip(x,y) if b == -1])

# Visualizamos los datos de entrenamiento, poniendo un color diferente a cada clase
scatter1=plt.scatter(positivos[:,1], positivos[:,2], c="lightgreen")
scatter2=plt.scatter(negativos[:,1], negativos[:,2], c="purple")

# Mostramos el nivel 0 de la proyección de la superficie obtenida en el plano z=0
plt.contour(xx,yy,z,[0],colors='aqua')
# Creamos un gráfico en 2D que no muestre nada para poder poner leyenda,
# ya que la leyenda no admite el tipo devuelto por la función contour
contorno_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='aqua', marker = '_')
plt.legend([scatter1, scatter2, contorno_proxy], ['Etiqueta 1','Etiqueta -1','Solución obtenida'], numpoints = 1, loc='upper left')
plt.show()

continuar()

print("Repetimos 1000 veces el experimento")
#Inicializamos a 0 las medias de los errores 
media_Ein=0
media_Eout=0

for i in range(1000):
    # Generamos los datos de entrenamiento, con el vector de características lineales
    xlin,y=datos()
    # Consideramos el siguiente vector de características no lineales
    x=np.array([[1,x1,x2,x1*x2,x1**2,x2**2] for x1,x2 in xlin[:,1:]],dtype='float64')
    # Estimamos los pesos con el algoritmo de sgd
    w=sgd(x,y,0.01,10000,32)
    # Añadimos el error en la muestra obtenido en esta iteración
    media_Ein+=Err(x,y,w)
    # Generamos los datos de test
    xlin_test,y_test=datos()
    x_test=np.array([[1,x1,x2,x1*x2,x1**2,x2**2] for x1,x2 in xlin_test[:,1:]],dtype='float64')
    # Añadimos el error en el conjunto de test obtenido en esta iteración
    media_Eout+=Err(x_test,y_test,w)

# Calculamos las medias de los errores 
media_Ein=media_Ein/1000; media_Eout=media_Eout/1000

print("Errores medios tras 1000 repeticiones del experimento con características no lineales")
print("Ein medio:",media_Ein)
print("Eout medio:",media_Eout)

continuar()

###############################################################################
###############################################################################
###############################################################################
print('BONUS: EJERCICIO SOBRE EL MÉTODO DE NEWTON\n')
print('Ejercicio 1\n')

# Retomamos la función f(x,y) definida en el ejercicio 3 de la búsqueda iterativa de óptimos,
# así como sus derivadas parciales y gradiente. 

#Función f(x,y)
def f(w):
    x,y=w[0],w[1]
    return (x+2)**2+2*(y-2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
#Derivada parcial de f con respecto a x
def fx(w):
    x,y=w[0],w[1]
    return 2*(x+2)+4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
#Derivada parcial de f con respecto a y
def fy(w):
    x,y=w[0],w[1]
    return 4*(y-2)+4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
#Gradiente de la función f
def gradf(w):
    return np.array([fx(w),fy(w)],dtype='float64')

#Definimos ahora las derivadas parciales de segundo orden
def fxx(w):
    x,y=w[0],w[1]
    return 2-8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
def fxy(w):
    x,y=w[0],w[1]
    return 8*(np.pi**2)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
def fyy(w):
    x,y=w[0],w[1]
    return 4-8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

# Calculamos la matriz Hessiana de f
def Hessf(w):
    # Nota: Como f es de clase 2, se cumple que fxy=fyx
    # y la matriz Hessiana es simétrica
    return np.array([[fxx(w),fxy(w)],[fxy(w),fyy(w)]],dtype='float64')

def Newton(init,f,grad,hess,maxIter):
    """ Función que implementa el algorimto de minimización de Newton
        y almacena en un vector las imágenes por la función f
        de los puntos obtenidos en cada iteración.

        Args:
            init: punto inicial del que parte el algoritmo
            f: función a la que se aplica el algoritmo
            grad: gradiente de la función f
            hess: matriz hessiana de la función f
            maxIter: máximo número de iteraciones del algoritmo

        Returns:
            w: punto obtenido tras aplicar el algoritmo 
            un número de maxIter iteraciones
            imagenes: vector de imágenes de la función f
            de los puntos obtenidos en cada iteración del algoritmo.  
    """
    #Añadimos en primer lugar al vector de imágenes el valor de la función en el punto inicial
    imagenes=np.array([f(init)],dtype='float64')
    #Inicializamos las coordenadas con el valor que se pasa como parámetro
    w=init
    #Iteramos hasta alcanzar maxIter
    for i in range(maxIter):
        # Determinamos la variación a aplicar al punto actual
        # np.linalg.inv calcula la inversa de una matriz
        # np.dot calcula el producto de la inversa de la matriz hessiana por el vector gradiente
        var=np.dot(np.linalg.inv(hess(w)),grad(w))
        # Actualizamos el punto
        w=w-var
        #Añadimos al vector de imagenes el valor de la función en el nuevo punto
        imagenes=np.append(imagenes,f(w))
    return w,imagenes
   
def gd(init,f,grad,maxIter,lr):
    """ Función que implementa el algorimto de gradiente descendente
        y almacena en un vector las imágenes por la función f
        de los puntos obtenidos en cada iteración.

        Args:
            init: punto inicial del que parte el algoritmo
            f: función a la que se aplica el algoritmo
            grad: gradiente de la función f
            maxIter: máximo número de iteraciones del algoritmo
            lr: learning rate (tasa de aprendizaje)

        Returns:
            w: punto obtenido tras aplicar el algoritmo 
            un número de maxIter iteraciones
            imagenes: vector de imágenes de la función f
            de los puntos obtenidos en cada iteración del algoritmo.  
    """
    #Añadimos en primer lugar al vector de imágenes el valor de la función en el punto inicial
    imagenes=np.array([f(init)],dtype='float64')
    #Inicializamos las coordenadas con el valor que se pasa como parámetro
    w=init
    #Iteramos hasta alcanzar maxIter
    for i in range(maxIter):
        #Actualizamos las coordenadas en la dirección del gradiente descendente
        w=w-lr*grad(w)
        #Añadimos al vector de valores el valor de la función en las nuevas coordenadas
        imagenes=np.append(imagenes,f(w))
    return w,imagenes

# Número máximo de iteraciones
maxIter=50
# Vector con los distintos puntos de inicio que vamos a usar para aplicar los dos métodos
init=np.array([[-1.0,1.0],[-2,2],[2.1, -2.1],[-3.0, 3.0],[-0.5, -0.5],[1.0, 1.0]],dtype='float64')

# Iteramos sobre los distintos puntos de inicio
for punto in init:
    # Aplicamos ambos métodos
    w_new,new=Newton(punto,f,gradf,Hessf,maxIter)
    w_gd1,gd1=gd(punto,f,gradf,maxIter,0.1)
    w_gd2,gd2=gd(punto,f,gradf,maxIter,0.01)
    

    # Imprimimos por pantalla los resultados
    print("Punto de inicio = ",punto)
    print("\nMétodo de Newton\n")
    print("Valor mínimo: ",f(w_new))
    print("Coordenadas donde se alcanza el valor mínimo:\n (", w_new[0],',',w_new[1],')')

    print("\nGradiente descendente con lr=0.1\n")
    print("Valor mínimo: ",f(w_gd1))
    print("Coordenadas donde se alcanza el valor mínimo:\n (", w_gd1[0],',',w_gd1[1],')')
    
    print("\nGradiente descendente con lr=0.01\n")
    print("Valor mínimo: ",f(w_gd2))
    print("Coordenadas donde se alcanza el valor mínimo:\n (", w_gd2[0],',',w_gd2[1],')')


    # Visualizamos la curva de decrecimiento de los valores de la función

    plt.figure(figsize=(8,8))
    plt.plot(range(maxIter+1),new,color='mediumspringgreen',marker='o',label='Método de Newton')
    plt.plot(range(maxIter+1),gd1,color='turquoise',marker='o',label='Gradiente Descendente\n lr=0.1')
    plt.plot(range(maxIter+1),gd2,color='orange',marker='o',label='Gradiente Descendente\n lr=0.01')
    plt.title(f'Comparación de los algoritmos\n Punto de inicio={punto}')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.legend()
    plt.show()
    
    continuar()

print("FIN DE LA PRÁCTICA 1")