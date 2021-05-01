# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Pilar Navarro Ramírez
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

# Fijamos la semilla
np.random.seed(1)

def continuar():
    input("\n----Presiona Enter para continuar----\n")

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
    a = (y2-y1)/(x2-x1) # Cálculo de la pendiente.
    b = y1 - a*x1       # Cálculo del término independiente.
    
    return a, b


#-----------EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente-----------

print("EJERCICIO 1.1")

unif = simula_unif(50, 2, [-50,50])
gaus = simula_gaus(50, 2, np.array([5,7]))

#Representamos gráficamente las nubes de puntos generadas
plt.figure(figsize=(6,6))
plt.scatter(unif[:,0],unif[:,1],color="turquoise")
plt.xlabel("Primera coordenada")
plt.ylabel("Segunda coordenada")
plt.title("Puntos generados aleatoriamente\n con distribución uniforme en [-50,50]x[-50,50]")
plt.show()

continuar()
plt.figure(figsize=(6,6))
plt.scatter(gaus[:,0],gaus[:,1],color="darkorange")
plt.xlabel("Primera coordenada")
plt.ylabel("Segunda coordenada")
plt.title("Puntos generados aleatoriamente\n con distribución normal  N((0,0), (sqrt(5), sqrt(7))) ")
plt.show()
continuar()

plt.figure(figsize=(6,6))
plt.scatter(gaus[:,0],gaus[:,1],color="darkorange",label="Distribución normal")
plt.scatter(unif[:,0],unif[:,1],color="turquoise",label="Distribución uniforme")
plt.xlabel("Primera coordenada")
plt.ylabel("Segunda coordenada")
plt.title("Puntos generados aleatoriamente con distintas distribuciones")
plt.legend()
plt.show()
continuar()

#---------- EJERCICIO 1.2: Generar nube de puntos y etiquetarlos con el signo de la distancia a una recta --------
print("EJERCICIO 1.2")
print("----Apartado a)-----")
#Generamos 100 puntos de dimensión 2 según una distribución uniforme en [-50,50]x[-50,50] 
x=simula_unif(100, 2, (-50,50))
# Generamos los parámetros de una recta
a, b = simula_recta([-50,+50])

print("Coeficientes de la recta usada para clasificar")
print("a: ",a, " b:",b)
continuar()

# Función signo 
def signo(x):
    if x >= 0:
        return 1
    return -1

#Función que determina la distancia de un punto a la recta y=ax+b
def f(x,y):
    return y-a*x-b


def etiquetar(x,f,ruido=False):
    """ Función que etiqueta los puntos de la muestra x, haciendo uso de la función f. 
        Además, añade ruido a las etiquetas si así lo indica el argumento ruido.  
        
        Args:
            x:puntos de la muestra a etiquetar
            f:función que determina la frontera de clasificación usada para etiquetar
            ruido: booleano que indica si se introduce ruido en el etiquetado o no
        Returns:
            etiquetas: vector con las etiquetas {1,-1} asignadas a cada punto de la
                        muestra x
        
        """
    #Asignamos una etiqueta a cada punto de la muesta x, según el signo de la función f
    etiquetas=np.array(list(map(lambda z: signo(f(z[0],z[1])), x)))
    if ruido:# Añadimos ruido a las etiquetas
        #Vector con las posiciones de los puntos que tienen etiqueta 1
        etiqueta1=np.array([i for i in range(len(etiquetas)) if etiquetas[i]==1])
        # Creamos un vector aleatorio de índices con un tamaño igual al 10% de las etiquetas positivas
        indices1=np.random.randint(len(etiqueta1),size=int(len(etiqueta1)*0.1))
        #Vector con las posiciones de los puntos que tienen etiqueta -1
        etiqueta2=np.array([i for i in range(len(etiquetas)) if etiquetas[i]==-1])
        # Creamos un vector aleatorio de índices con un tamaño igual al 10% de las etiquetas negativas
        indices2=np.random.randint(len(etiqueta2),size=int(len(etiqueta2)*0.1))
        for i in indices1:
            #Cambiamos el signo de las etiquetas positivas cuyas posiciones se encuentran en indices1
            etiquetas[etiqueta1[i]]=-etiquetas[etiqueta1[i]]
        for i in indices2:
            #Cambiamos el signo de las etiquetas negativas cuyas posiciones se encuentran en indices2
            etiquetas[etiqueta2[i]]=-etiquetas[etiqueta2[i]]       
    return etiquetas

def plotEtiquetado(x,etiquetas,f,title,axis_labels,legend_labels,lim=False):
    """ Función para visualizar una muesta etiquetada, junto con el etiquetado generado 
        por el clasificador determinado por la función f 
        
        Args: 
            x: muestra de puntos a visualizar
            etiquetas: vector de etiquetas asociado a la muestra x
            f: función que determina una frontera de clasificación
            title: título para el gráfico generado 
            axis_labels: lista de etiquetas para los ejes
            legend_labels: lista de etiquetas para la leyenda
            lim: booleano que determina si se establecen límites a los valores de los ejes
        """
    
    plt.figure(figsize=(7,7))
    # Colores para las regiones positiva y negativa
    cm = ListedColormap(['thistle','azure'])

    #Creamos una malla de puntos equiespaciados donde evaluar la función f 
    xx = np.linspace(-50, 50, 100, endpoint=True)
    yy = np.linspace(-50, 50, 100, endpoint=True)
    X, Y = np.meshgrid(xx,yy)
    #Evaluamos la función en los puntos de la malla
    Z = f(X,Y)
    
    if lim: # Si el parámetro lim es True
        #Determinamos los extremos de los valores de las características
        min_x1= np.min(x[:,0])
        max_x1 = np.max(x[:,0])
        min_x2= np.min(x[:,1])
        max_x2 = np.max(x[:,1])

        # Establecemos los límites de los ejes
        plt.xlim(min_x1,max_x1)
        plt.ylim(min_x2,max_x2)
    
    #Ponemos etiquetas a los ejes
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    
    # Mostramos el nivel 0 de la proyección del clasificador en el plano z=0
    plt.contour(X,Y,Z,[0],colors='black') #Para mostrar el borde del clasificador
    plt.contourf(X,Y,Z,0,cmap=cm,alpha=0.4) # Para mostrar las regiones de clasificación rellenas

    # Creamos dos arrays, cada uno con los datos de la muestra de una clase
    positivos = np.array([a for a,b in zip(x,etiquetas) if b == 1])
    negativos = np.array([a for a,b in zip(x,etiquetas) if b == -1])

    # Visualizamos los datos, poniendo un color diferente a cada clase
    scatter1=plt.scatter(positivos[:,0], positivos[:,1], c="limegreen",label=legend_labels[0],alpha=0.75)
    scatter2=plt.scatter(negativos[:,0], negativos[:,1], c="darkorchid",label=legend_labels[1],alpha=0.75)

    # Creamos un gráfico en 2D que no muestre nada para poder poner leyenda,
    # ya que la leyenda no admite el tipo devuelto por la función contour
    line_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='black', marker = '_')
    pos_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='azure', marker = 's')
    neg_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='thistle', marker = 's')
    
    plt.legend([scatter1,scatter2,line_proxy, pos_proxy, neg_proxy], [scatter1.get_label(),scatter2.get_label(), 
            legend_labels[2],legend_labels[3],legend_labels[4]], numpoints = 1,framealpha=0.5)
    plt.title(title)
    plt.show()
  
#Asigno etiquetas sin ruido a la muestra x
etiquetas=etiquetar(x,f)
#Visualizamos el etiquetado resultante junto con la recta de clasificación
plotEtiquetado(x,etiquetas,f,"Muestra etiquetada sin ruido usando una recta",["Primera coordenada","Segunda coordenada"],
              ["Etiqueta +1","Etiqueta -1","f(x,y)=0","f(x,y)>0","f(x,y)<0"])
continuar()

print("----Apartado b)----")

#Función para calcular el error en la clasificación
def Error(x, etiquetas, f):
    """ Función que calcula la proporción de puntos mal clasificados
        por el clasificador determinado por la función f
    
    Args:
        x: puntos de una muestra
        etiquetas: vector de etiquetas asociadas a la muestra
        f: función que determina una frontera de clasificación
        
    Returns: proporción de puntos de la muestra x que el clasificador f 
            ha clasificado incorrectamente. 
    
    """
    error = 0
    for i in range(0, len(x)):
        #Calculamos el error total como el número de elementos mal clasificados
        if signo(f(x[i][0], x[i][1]))!= etiquetas[i]:
            error = error + 1
    return error/len(x)

# Asigno etiquetas a la muestra x con ruido en el 10% de cada clase
etiquetas_ruido=etiquetar(x,f,True)
print("Proporción de puntos mal clasificados por la recta al incluir ruido:", Error(x,etiquetas_ruido,f))
# Visualizamos el etiquetado resultante junto con la recta de clasificación
plotEtiquetado(x,etiquetas_ruido,f,
               "Muestra etiquetada usando una recta\n con ruido en el 10% de las etiquetas de cada clase",
               ["Primera coordenada","Segunda coordenada"], ["Etiqueta +1","Etiqueta -1","f(x,y)=0","f(x,y)>0","f(x,y)<0"])
continuar()

# Apartado c): Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta
print("----Apartado c)-----")

#Nuevas fronteras de clasificación
def f1(x,y):
    return (x-10)**2+(y-20)**2-400
def f2(x,y):
    return 0.5*(x+10)**2+(y-20)**2-400
def f3(x,y):
    return 0.5*(x-10)**2-(y+20)**2-400
def f4(x,y):
    return y-20*x**2-5*x+3

#Creamos un vector con las funciones que determinan las fronteras de clasificación
fs=[f1,f2,f3,f4]
# Vector de cadenas de caracteres para describir a las fronteras de clasificación
f_str=['(x-10)^2+(y-20)^2-400','0.5(x+10)^2+(y-20)^2-400','0.5(x-10)^2-(y+20)^2-400','y-20x^2-5x+3']

for i in range(len(fs)):
    print("Frontera de clasificación f(x,y)="+ f_str[i])
    etiquetas_nuevas=etiquetar(x,fs[i],True)
    print("Proporción de puntos mal clasificados:", Error(x,etiquetas_nuevas,fs[i]))
    plotEtiquetado(x,etiquetas_nuevas,fs[i],
                   "Muestra etiquetada con ruido en el 10% de las etiquetas de cada clase\n Frontera de clasificación f(x,y)="+f_str[i],
                   ["Primera coordenada","Segunda coordenada"], ["Etiqueta +1","Etiqueta -1","f(x,y)=0","f(x,y)>0","f(x,y)<0"])
    continuar()

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

#Función para calcular el error en la clasificación
def Error(x, etiquetas, w):
    """ Función que calcula la proporción de puntos mal clasificados
        por el hiperplano determinado por el vector de pesos w
    
    Args:
        x: puntos de una muestra
        etiquetas: vector de etiquetas asociadas a la muestra
        w: vector de pesos que determina un hiperplano
        
    Returns: proporción de puntos de la muestra x clasificados incorrectamente
            por el hiperplano determinado por w. 
    
    """
    error = 0
    for i in range(0, len(x)):
        #Calculamos el error total como el número de elementos mal clasificados
        if signo(np.dot(w,x[i]))!= etiquetas[i]:
            error = error + 1
    return error/len(x)


def ajusta_PLA(datos, label, max_iter, vini):
    """ Función que implementa el algoritmo PLA
    
    Args: 
        datos: matriz de características
        label: vector de etiquetas asociadas a cada fila de la matriz datos
        max_iter: máximo número de iteraciones permitidas del algoritmo
        vini: vector inicial de pesos del que parte el algoritmo
        
    Returns:
        iterations: número de iteraciones necesarias hasta encontrar la solución óptima.
                    Será igual a max_iter si los datos no son linealmente separables. 
        w: vector de pesos que determina el hiperplano solución
    
    """
    #Inicializamos el vector de pesos con el valor que se pasa como parámetro
    w = vini 
    #Booleano para determinar si se produce cambio en los pesos en una iteración del algoritmo
    change = True     
    iterations = 0 # Guardamos el número de iteraciones
    # Iteramos mientras haya datos mal clasificados (se produzca cambio en el vector de pesos)
    # y no se haya superado el número máximo de iteraciones permitidas
    while change and iterations < max_iter:
        change = False      
        #Iteramos sobre todos los vectores de características (filas de la matriz datos)
        for i, x in enumerate(datos): 
            # Si el punto está mal clasificado, es decir, si el signo del producto escalar 
            # del vector de pesos y el vector de características no coincide con 
            # la etiqueta asociada a ese vector, se actualizan los pesos
            if signo(np.dot(w,x)) != label[i]: 
                w = w + label[i]*x
                change = True
        iterations += 1 #Actualizamos el número de iteraciones
    return w, iterations


print("EJERCICIO 2.1")

def iteracionesPLA(x, etiquetas):
    datos=np.array([[1,x1,x2] for x1,x2 in x],dtype='float64') #Añadimos un 1 a cada vector de características
    
    vini=np.zeros(datos.shape[1]) #Inicializamos el vector de pesos al vector 0
    #Llamamos al algoritmo PLA partiendo del vector 0
    w,iterations=ajusta_PLA(datos,etiquetas,10000,vini)

    print("Número de iteraciones necesarias para converger del algoritmo PLA partiendo del vector cero: ", iterations)
    print("Solución obtenida: ",w)
    print("Proporción de puntos mal clasificados:",Error(datos,etiquetas,w))

    #Visualizamos la recta de clasificación obtenida junto con la muestra etiquetada
    label=('{0:+}'.format(-w[1]/w[2]))[:8]+'x'+('{0:+}'.format(-w[0]/w[2]))[:8]
    plotEtiquetado(x,etiquetas,lambda x,y: w[0] + w[1]*x + w[2]*y,
                   "Muestra etiquetada y solución obtenida con PLA\n partiendo del vector cero",
                   ["Primera coordenada","Segunda coordenada"], ["Etiqueta +1","Etiqueta -1","y="+label,"y>"+label,"y<"+label])
    continuar()

    print("Partiendo de vectores de números aleatorios en [0,1]\n")
    media_iterations=0  # Para guardar la media del número de iteraciones necesarias para que converja el algoritmo
    media_error=0 #Para guardar la media del error
    for i in range(10): # Llamamos 10 veces al algoritmo PLA partiendo en cada ejecución de un vector aleatorio diferente
        #Inicializamos el vector de pesos con números aleatorios en el intervalo [0,1]
        vini=np.random.rand(datos.shape[1]) 
        w,iterations=ajusta_PLA(datos,etiquetas,10000,vini)
        print("Vector inicial de  pesos: [",vini[0],",",vini[1],",",vini[2],"]")
        print("Número de iteraciones necesarias: ", iterations)
        error=Error(datos,etiquetas,w)
        print("Proporción de puntos mal clasificados:",error)
        media_error+=error
        media_iterations+=iterations
    media_iterations/=10
    media_error/=10

    print("\nNúmero medio de iteraciones necesarias para converger del algoritmo PLA\npartiendo de vectores de números aleatorios en [0,1]: ", media_iterations)
    print("Error medio:", media_error)
    print("\nSolución obtenida en la última iteración: ", w)

    #Visualizamos la recta de clasificación obtenida junto con la muestra etiquetada
    label=('{0:+}'.format(-w[1]/w[2]))[:8]+'x'+('{0:+}'.format(-w[0]/w[2]))[:8]
    plotEtiquetado(x,etiquetas,lambda x,y: w[0] + w[1]*x + w[2]*y,
                   "Muestra etiquetada y solución obtenida con PLA\n partiendo de un vector aleatorio en [0,1]",
                   ["Primera coordenada","Segunda coordenada"],["Etiqueta +1","Etiqueta -1","y="+label,"y>"+label,"y<"+label])

print("\n---- Apartado 1): Datos linealmente separables-------\n")
iteracionesPLA(x,etiquetas)
continuar()

print("\n---- Apartado 2): Datos con ruido (no separables) -------\n")
iteracionesPLA(x,etiquetas_ruido)
continuar()

# EJERCICIO 2.2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
print("EJERCICIO 2.2")

# Función sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Error de regresión logística en la muestra
def Error_LR(w,x,y):
    return np.mean(np.log(1+np.exp(-y*x.dot(w).T)))

# Gradiente del error en el punto (x,y)
def gradError(w, x, y):
    return -y*x*sigmoid(-y*np.dot(w,x))


# Gradiente Descendente Estocástico para regresión logística
def sgdLR(x,y, maxEpocas, lr=0.01):
    """ Función que implementa el algoritmo de gradiente descendente estocástico
     en un modelo de regresión logística con tamaño de batch de 1

    Args:
        x: matriz con las características de cada dato
        y: etiquetas asociadas a cada dato
        maxEpocas: número máximo de épocas
        lr: learning rate (tasa de aprendizaje)

    Returns:
        w: vector de pesos obtenido por el algoritmo
        epocas: número de épocas que necesita el algoritmo para converger.
    """
    #Inicializamos los pesos como un vector de ceros
    #con una longitud igual a la de cada vector de caracaterísticas x_i
    w = np.zeros(x.shape[1],dtype='float64')
    #Booleano para determinar si se produce un cambio significativo en los pesos en una época del algoritmo
    change=True 
    epocas=0
    #Iteramos mientras el número de épocas recorridas sea menor que el máximo número de épocas permitido
    # y se produzca una modificación significativa en el vector de pesos
    while epocas<maxEpocas and change :
        w_anterior=w.copy() #Guardamos el vector de pesos de la época anterior
        # Consideramos un vector de índices permutados aleatoriamente,
        # con tantos índices como elementos hay en x (en nuestro dataset)
        indices=np.random.permutation(x.shape[0])
        for i in indices: 
            #Actualizamos los pesos en la dirección del gradiente descendente del error en cada dato (x,y) de la muestra
            w=w-lr*gradError(w,x[i],y[i]) 
        # Vemos si la diferencia en norma del vector de pesos obtenido en la época anterior
        # y el obtenido en la época actual es considerable
        change=np.linalg.norm(w_anterior-w)>0.01 
        epocas+=1
    return w,epocas


# Generamos una muestra de 100 puntos uniformememte distribuidos en [0,2]x[0,2]
x=simula_unif(100,2,[0,2])
# Añadimos un 1 a cada vector de características
x=np.array([[1,x1,x2] for x1,x2 in x],dtype='float64') 
#Generamos una recta aleatoria, que corta al cuadrado [0,2]x[0,2], como frontera 
a,b = simula_recta([0,2])
print("Coeficientes de la recta usada para clasificar")
print("a: ",a, " b:",b)
continuar()
# Etiquetamos la muestra haciendo uso de la recta generada
etiquetas = etiquetar(x[:,1:],f)
# Visualizamos el etiquetado resultante junto con la recta de clasificación
label=('{0:+}'.format(a))[:8]+'x'+('{0:+}'.format(b))[:8]
plotEtiquetado(x[:,1:],etiquetas,f,"Muestra etiquetada usando una recta aleatoria",["Primera coordenada","Segunda coordenada"],
               ["Etiqueta +1","Etiqueta -1","y="+label,"y>"+label,"y<"+label],True)
continuar()
print("Regresión logística con SGD\n")
#Aplicamos el algoritmo de SGD para regresión logística para buscar la solución
w,epocas=sgdLR(x,etiquetas,10000)
print("Vector de pesos obtenido: ", w)
print("Número de épocas necesario para converger:", epocas)
print("Error de regresión logística en la muestra (Ein):", Error_LR(w,x,etiquetas))
print("Proporción de puntos mal clasificados en la muestra:", Error(x,etiquetas,w))
continuar()
#Visualizamos la recta obtenida junto con la muestra etiquetada
label=('{0:+}'.format(-w[1]/w[2]))[:8]+'x'+('{0:+}'.format(-w[0]/w[2]))[:8]
plotEtiquetado(x[:,1:],etiquetas,lambda x,y: w[0] + w[1]*x + w[2]*y,"Muestra etiquetada y solución obtenida usando regresión logística",
               ["Primera coordenada","Segunda coordenada"],["Etiqueta +1","Etiqueta -1","y="+label,"y>"+label,"y<"+label],True)
continuar()
#Generamos el conjunto de test
test=simula_unif(1500,2,[0,2])
test=np.array([[1,x1,x2] for x1,x2 in test],dtype='float64') 
etiquetas_test = etiquetar(test[:,1:],f)

print("Error de regresión logística fuera de la muestra (Eout):", Error_LR(w,test,etiquetas_test))
print("Proporción de puntos mal clasificados en el conjunto de test:", Error(test,etiquetas_test,w))
continuar()

print("\nRepetimos 100 veces el experimento\n")

#Inicializamos a 0 las medias
media_epocas=0 
media_Eout=0
media_Ein=0
media_error_in=0
media_error_out=0

for i in range(100):
    # Generamos una muestra de 100 puntos uniformememte distribuidos en [0,2]x[0,2]
    x=simula_unif(100,2,[0,2])
    # Añadimos un 1 a cada vector de características
    x=np.array([[1,x1,x2] for x1,x2 in x],dtype='float64') 
    #Generamos una recta aleatoria, que corta al cuadrado [0,2]x[0,2], como frontera 
    a,b = simula_recta([0,2])
    # Etiquetamos la muestra haciendo uso de la recta generada
    etiquetas = etiquetar(x[:,1:],f)
    #Aplicamos el algoritmo de SGD para regresión logística para buscar la solución
    w,epocas=sgdLR(x,etiquetas,10000)
    media_epocas+=epocas #Añadimos las épocas necesarias para que converja el algoritmo en esta iteración
    media_Ein+=Error_LR(w,x,etiquetas) # Añadimos el error en la muestra obtenido en esta iteración
    media_error_in+=Error(x,etiquetas,w)
    #Generamos el conjunto de test
    test=simula_unif(1500,2,[0,2])
    test=np.array([[1,x1,x2] for x1,x2 in test],dtype='float64') 
    etiquetas_test = etiquetar(test[:,1:],f)
    # Añadimos el error en el conjunto de test obtenido en esta iteración
    media_Eout+=Error_LR(w,test,etiquetas_test) 
    media_error_out+=Error(test,etiquetas_test,w) 

print("Ein medio tras 100 repeticiones del experimento:",media_Ein/100)
print("Eout medio tras 100 repeticiones del experimento:",media_Eout/100)
print("Proporción media de puntos mal clasificados en la muestra tras 100 repeticiones:", media_error_in/100)
print("Proporción media de puntos mal clasificados en el conjunto de test tras 100 repeticiones:", media_error_out/100)
print("Número de épocas promedio que necesita el algoritmo de RL para converger:", media_epocas/100)
continuar()

#BONUS: Clasificación de Dígitos
print('BONUS: CLASIFICACIÓN DE DÍGITOS\n')
print('Ejercicio 2\n')

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
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='turquoise', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='darkorange', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='turquoise', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='darkorange', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

continuar()

print("Tamaño del conjunto de entrenamiento: ",len(y))
print("Tamaño del conjunto de test: ", len(y_test))

continuar()
#LINEAR REGRESSION FOR CLASSIFICATION 

# Algoritmo de la Pseudo-inversa
def pseudoinverse(x,y):
    #np.linalg.pinv calcula la matriz pseudo-inversa de la matriz x
    #np.dot computa el producto de la matriz pseudo-inversa de x por y
    return np.dot(np.linalg.pinv(x),y)


#POCKET ALGORITHM

def Pocket(datos, label, maxIter, vini):
    """ Función que implementa el algoritmo PLA-pocket 
    
    Args: 
        datos: matriz de características
        label: vector de etiquetas asociadas a cada fila de la matriz datos
        maxIter: máximo número de iteraciones permitidas del algoritmo
        vini: vector inicial de pesos del que parte el algoritmo
        
    Returns:
        mejor_error: menor error en la muestra encontrado 
                    por el algoritmo en maxIter iteraciones
        mejor_w: vector de pesos que determina el hiperplano solución
                con menor error en la muestra encontrado     
    
    """
    #Inicializamos el vector de pesos con el valor que se pasa como parámetro
    w = vini 
    mejor_w=w #Guardamos la mejor solución obtenida hasta el momento
    mejor_error=Error(datos,label,mejor_w)  #Guardamos el menor error obtenido hasta el momento
    for i in range(maxIter):
        #Iteramos sobre todos los vectores de características (filas de la matriz datos)
        for i, x in enumerate(datos): 
            # Si el punto está mal clasificado, es decir, si el signo del producto escalar 
            # del vector de pesos y el vector de características no coincide con 
            # la etiqueta asociada a ese vector, se actualizan los pesos
            if signo(np.dot(w,x)) != label[i]: 
                w = w + label[i]*x
        error=Error(datos,label,w) #Caculamos el error asociado al vector de pesos obtenido
        # Si el error del nuevo vector de pesos es menor que el mejor error obtenido hasta el momento,
        # se actualiza el valor del mejor w y mejor error
        if error<mejor_error: 
            mejor_error=error
            mejor_w=w
    return mejor_w, mejor_error 

#Calculamos la solución que nos da regresión lineal
w=pseudoinverse(x,y)
print("Solución obtenida con el algoritmo de la pseudoinversa:",w)
label=('{0:+}'.format(-w[1]/w[2]))[:8]+'x'+('{0:+}'.format(-w[0]/w[2]))[:8]
plotEtiquetado(x[:,1:],y,lambda x,y: w[0] + w[1]*x + w[2]*y,"Datos de entrenamiento\ny solución obtenida usando regresión lineal",
               ["Intensidad promedio","Simetría"],["Dígito 8","Dígito 4","y="+label,"y>"+label,"y<"+label],True)
continuar()
plotEtiquetado(x_test[:,1:],y_test,lambda x,y: w[0] + w[1]*x + w[2]*y,"Datos de test\ny solución obtenida usando regresión lineal",
                ["Intensidad promedio","Simetría"],["Dígito 8","Dígito 4","y="+label,"y>"+label,"y<"+label],True)
continuar()
print("Error en la muestra (E_in):", Error(x,y,w))
print("Error sobre los datos de test (E_test): ", Error(x_test,y_test,w))
continuar()

#Aplicamos ahora el algoritmo de PLA-Pocket como mejora
w_mejor,error=Pocket(x,y,10000,w)
print("Solución mejorada con el algoritmo de PLA-Pocket:",w_mejor)
label=('{0:+}'.format(-w_mejor[1]/w_mejor[2]))[:8]+'x'+('{0:+}'.format(-w_mejor[0]/w_mejor[2]))[:8]
plotEtiquetado(x[:,1:],y,lambda x,y: w_mejor[0] + w_mejor[1]*x + w_mejor[2]*y,"Datos de entrenamiento\ny solución obtenida usando PLA-Pocket",
                ["Intensidad promedio","Simetría"],["Dígito 8","Dígito 4","y="+label,"y>"+label,"y<"+label],True)
continuar()
plotEtiquetado(x_test[:,1:],y_test,lambda x,y: w_mejor[0] + w_mejor[1]*x + w_mejor[2]*y,"Datos de test\ny solución obtenida usando PLA-Pocket",
                ["Intensidad promedio","Simetría"],["Dígito 8","Dígito 4","y="+label,"y>"+label,"y<"+label],True)
continuar()
print("Error en la muestra (E_in):", error)
print("Error sobre los datos de test (E_test): ", Error(x_test,y_test,w_mejor))
continuar()
print("FIN DE LA PRÁCTICA 2")