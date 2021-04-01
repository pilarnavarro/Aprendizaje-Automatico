# Práctica 0 Pilar Navarro Ramírez

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Parte 1
#Cargamos el conjunto de datos de Iris de Scikit-Learn
data = load_iris()
#Nos quedamos con las características y las etiquetas
X=np.array(data.data)
Y=np.array(data.target)
print("-----------------PARTE 1------------------")
print(f"Características de las flores:\n {data.feature_names}\n {X}")
print(f"Clases:\n {data.target_names}\n{Y}")
input("Presiona Enter para continuar")
#Seleccionamos la primera y tercera característica
X_1_3 = X[:,0:3:2]
print(f"Características 1 y 3:\n{data.feature_names[0:3:2]}\n{X_1_3}")
input("Presiona Enter para continuar")

#Scatter Plot 
#Me he basado en el código que aparece en
#https://stackoverflow.com/questions/64068419/how-to-visualize-the-iris-dataset-on-2d-plots-for-different-combinations-of-feat

n_samples = len(Y) #Número de ejemplos en el conjunto de datos
plt.figure(figsize=(15,10)) #Para que la gráfica tenga mayor tamaño y se pueda ver mejor
for clase in set(Y): #Cogemos las diferentes etiquetas que aparecen en el conjunto de etiquetas
    x = [X[i,0] for i in range(n_samples) if Y[i]==clase] #Conjunto con las características 1 de los datos que pertenecen a la clase 'clase'
    y = [X[i,2] for i in range(n_samples) if Y[i]==clase] #Conjunto con las características 3 de los datos que pertenecen a la clase 'clase'
    # Creamos el gráfico con las características 1 y 3 de los datos
    # asignando un color a cada clase y el nombre correspondiente a dicha clase aparece en la leyenda
    plt.scatter(x, y, color=['orange', 'black', 'green'][clase], label=data.target_names[clase])
    plt.xlabel(data.feature_names[0]) #Nombre de la característica 1 para que aparezca en el eje x
    plt.ylabel(data.feature_names[2]) #Nombre de la característica 3 para que aparezca en el eje y
    plt.title('Iris Dataset')
    #Colocamos la leyenda con el nombre de las clases en la esquina inferior derecha
    plt.legend(data.target_names, loc='lower right') 
plt.show()

input("Presiona Enter para continuar")

# Parte 2
print("-----------------PARTE 2------------------")
#Partimos el conjunto de datos en conjunto de entrenamiento y conjunto de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, stratify=Y)

print("Cantidad de cada clase de flor en cada conjunto (train y test)")
print(f"Entrenamiento: {Counter(Y_train)} Test: {Counter(Y_test)}")
print("Tamaño del conjunto original:",len(X))
print("¿Supone el conjunto de entrenamiento el 75% del conjunto original?")
print(f"0.75 * 150 = {0.75*150}")
print("Tamaño del conjunto de entrenamiento:",len(X_train))
print("¿Supone el conjunto de test el 25% del conjunto original?")
print(f"0.25 * 150 = {0.25*150}")
print("Tamaño del conjunto de test:",len(X_test))

input("Presiona Enter para continuar")

# Parte 3
print("-----------------PARTE 3------------------")
#Creamos un array con 100 valores equiespaciados entre 0 y 4*pi
data=np.linspace(0, 4*np.pi, 100)
print("Valores equiespaciados:\n",data)

input("Presiona Enter para continuar")
#Creamos una array de arrays donde cada uno de ellos contiene los resultados de aplicar una función trigonométrica
#al array de 100 valores recién creado
trig = np.array([np.sin(data), np.cos(data), np.tanh(np.sin(data)+np.cos(data))])

#Visualización de las curvas correspondientes a las tres funciones trigonométricas
leyenda=['sen(x)','cos(x)','tanh(sen(x)+cos(x))']
plt.figure(figsize=(15, 10))
plt.plot(data,trig[0], linestyle='dashed', color='green', label=leyenda[0])
plt.plot(data,trig[1], linestyle='dashed', color='black', label=leyenda[1])
plt.plot(data,trig[2], linestyle='dashed', color='red', label=leyenda[2])
plt.title('Algunas funciones trigonométricas')
plt.legend(leyenda, loc='lower right')
plt.show()

print("Fin del programa")