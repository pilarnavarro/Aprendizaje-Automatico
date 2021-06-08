# -*- coding: utf-8 -*-
"""
TRABAJO 3 - Regresión
Nombre Estudiante: Pilar Navarro Ramírez
"""
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVR
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import model_selection
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

# Fijamos la semilla
np.random.seed(35)

def continuar():
    input("\n----Presiona Enter para continuar----\n")
    
# Funcion para leer los datos
def readData(file):
        data=pd.read_csv(file,sep=',',header=0) #leemos el archivo en un dataframe de pandas
        #print(data.describe())
        x=data.iloc[:,:-1] #Características
        y=data.iloc[:,-1] # Etiquetas
                
        return np.array(x), np.array(y)
    
print("--------------------PROBLEMA DE REGRESIÓN-------------------\n")
#Leemos los datos
x,y=readData("datos/train.csv")
print("Número de ejemplos en el conjunto de datos: ",len(x))
print("Tamaño de los vectores de características: ", x.shape[1])
continuar()

# Partimos el conjunto de datos en conjunto de entrenamiento y conjunto de test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

#Nos guardamos una copia de los datos de entrenamiento originales
x_train_orig=x_train.copy()
y_train_orig=y_train.copy()

print("Tamaño del conjunto de entrenamiento: ",len(x_train))
print("Tamaño del conjunto de test: ",len(x_test))

continuar()


def plot2D(x,y,title):
    """ Función para visualizar una muesta etiquetada en 2D

        Args: 
            x: muestra de puntos a visualizar
            y: vector de etiquetas asociado a la muestra x
            title: título para el gráfico generado 
        """

    plt.figure(figsize=(10,10))
    plt.scatter(x[:,0],x[:,1],c=y, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.title(title)
    plt.show()

    
print("\nRepresentación de los datos\n")

#Escalamos las características para que tengan media 0 y varianza 1
x_train = StandardScaler().fit_transform(x_train)


#Me he guiado por los siguientes tutoriales para entender los algoritmos PCA y TSNE y cómo usarlos
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

#Realizamos un análisis de componentes principales, para quedarnos con 2 componentes y poder visualizar los datos
pca=PCA(n_components=2, random_state=1)
x_pca=pca.fit_transform(x_train)

#Visualizamos los datos resultantes en 2D
plot2D(x_pca, y_train, 'PCA')

#Vemos la varianza explicada por cada una de las dos componentes
print("Varianza explicada: ", pca.explained_variance_ratio_)
continuar()

#Reducimos ahora la dimensionalidad con t-SNE, partiendo de los resultados obtenidos con pca
x_tsne = TSNE(n_components=2, init=x_pca).fit_transform(x_train)

#Visualizamos los datos resultantes en 2D
plot2D(x_tsne, y_train, 'TSNE')

continuar()
from sklearn.neighbors import LocalOutlierFactor
# Función para visualizar una matriz
def plotMatrix(matrix, title, labels=False):
    """ Función para visualizar una matriz como un mapa de calor

    Args: 
        matrix: matriz a visualizar
        title: título para el gráfico generado 
        labels: booleano que determina si se añaden etiquetas a los ejes
    """


    plt.figure(figsize=(8,8))
    sb.heatmap(matrix, cmap='viridis')
    plt.title(title,fontsize=15)
    
    if labels:
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
    plt.show()

def dependence(x,y,threashold):
    """ Función que determina los índices de las características de las que más depende 
        el atributo objetivo, o, lo que es lo mismo, que aportan más información
        
        Ags:
            x:conjunto de características que se usan para la predicción
            y: conjunto de etiquetas (valores del atributo objetivo) asociadas
                a cada una de las instancias de x
            threashold: umbral que indica el valor de MI (mutal information)
                        mínimo para elegir una característica
        Returns:
             vector de índices correspondientes a las características
             que aportan más información, es decir, cuya dependencia 
             con el atributo objetivo es mayor que threashold.           
    """
    m = list(enumerate(mutual_info_regression(x,y,random_state=24)))
    best = []
    for(i,info) in m:
        if info > threashold:
                best.append(i)
    return np.array(best) 

def removeFeatures(x,index):
    """ 
    Función que devuelve un nuevo conjunto de características, 
    con las características de x cuyas posiciones
    se encuentran en el vector index 
    """
    
    new=[[] for i in range(x.shape[0])]
    for i in index:
        for j in range(x.shape[0]):
                new[j].append(x[j,i])
    return np.array(new)
    
def detectOutliers(x):
    """
        Función para detectar los outliers 
        presentes en el conjunto de datos x. 
        Devuelve un vector de índices con las posiciones
        de los outliers en el dataset x.     
    """
    
    #Definimos el LOF
    detector=LocalOutlierFactor(n_neighbors=20,contamination='auto')
    #Ajustamos el detector de outliers al conjunto de datos y predecimos las etiquetas 
    # (1 para inlier y -1 para outlier)
    results=detector.fit_predict(x)
    outliers=[]
    for i in range(len(results)):
        if results[i]==-1:  #Si es un outliers tendrá etiqueta -1
            outliers.append(i) #Guardamos los índices de los outliers
    return np.array(outliers)

def removeExamples(x,index):
    """ 
    Función que devuelve un nuevo conjunto de ejemplos, 
    con las instancias del conjunto x cuyas posiciones 
    no aparecen en el vector index.
    """
    
    new=[]
    for i in range(x.shape[0]):
        if i not in index:
            if len(x.shape)!=1:
                new.append(x[i][:])
            else:
                new.append(x[i])
    return np.array(new)

print("\nPREPROCESAMIENTO\n")

#Escalamos las características para que tengan media 0 y varianza 1
scaler = StandardScaler()

#Recuperamos los datos de entrenamiento originales
x_train=x_train_orig.copy()
y_train=y_train_orig.copy()

scaler.fit(x_train) #Normalizamos los datos
x_train_prep=scaler.transform(x_train) 
x_test_prep=scaler.transform(x_test)

#Eliminamos los outliers
outliers=detectOutliers(x_train_prep)
x_train_prep=removeExamples(x_train_prep,outliers)
y_train=removeExamples(y_train,outliers)

#Nos quedamos con las características que aportan más información
#las que tienen un valor de mutual information (MI) mayor a 0.4
index=dependence(x_train_prep,y_train,0.4)
x_train_prep=removeFeatures(x_train_prep,index)
x_test_prep=removeFeatures(x_test_prep,index)


print("Nuevo número de características: ", len(x_train_prep[0]))
print("Nuevo número de instancias: ", len(x_train_prep))
continuar()

#Realizamos un análisis de componentes principales, para poder visualizar los nuevos datos
pca=PCA(n_components=2, random_state=1)
x_pca=pca.fit_transform(x_train_prep)

#Visualizamos los datos resultantes en 2D
plot2D(x_pca, y_train, 'PCA')

#Vemos la varianza explicada por cada una de las dos componentes
print("Varianza explicada: ", pca.explained_variance_ratio_)
continuar()

#Reducimos ahora la dimensionalidad con t-SNE, partiendo de los resultados obtenidos con pca
x_tsne = TSNE(n_components=2, init=x_pca,perplexity=30).fit_transform(x_train_prep)

#Visualizamos los datos resultantes en 2D
plot2D(x_tsne, y_train, 'TSNE')

continuar()

#Calculamos la matriz de correlación entre las características, usando el coeficiente de Pearson
#np.corrcoef asume que cada fila de la matriz pasada como parámetro es una característica y cada
#columna es una observación, por lo que hay que trasponer la matriz de características
corr = np.corrcoef(np.transpose(x_train_prep))

#Visualizamos la matriz de correlación obtenida
plotMatrix(corr,"Matriz de correlación usando coeficiente de Pearson\ntras el preprocesamiento")

TUNING=False #Parámetro que indica si se ejecuta o no la configuración de parámetros

def printCV(x, y, reg, modelo, grid):
    """ Función que imprime por pantalla los resultados 
        de validación cruzada obtenidos por el modelo indicado, 
        sobre los datos que se pasan como parámetro 
        
        Args: 
            x:conjunto de datos de entrenamiento 
            y:vector de etiquetas asociadas a x
            reg: modelo cuya calidad se quiere medir
            modelo: nombre del modelo 
            grid: booleano que indica si se ha llamado a la función después de ajustar los
                  parámetros con un grid search
        
        """
    
    print(modelo)
    
    if grid:
        print("Mejores parámetros:\n", reg.best_params_) 
        print("Error de validación cruzada del mejor estimador: ",  -reg.best_score_ )
    else:
        #Validación cruzada con k=5 para estimar el error fuera de la muestra
        E_val=model_selection.cross_val_score(reg, x, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        print("Error de validación cruzada: ",  -np.mean(E_val))
    continuar()
    
def printFinalResults(x_test, x_train, reg, modelo):
    """
    Función que imprime por pantalla los resultados en el conjunto
    de test y conjunto de entrenamiento del modelo indicado
        
    Args: 
        x_train: conjunto de datos preprocesados, para el entrenamiento del modelo
        x_test: conjunto de datos de test preprocesados
        reg: modelo cuya calidad se quiere medir
        modelo: nombre del modelo    
    """
    
    print(modelo)
    #Entrenamos el modelo
    reg.fit(x_train,y_train)
    print("Error en el conjunto de entrenamiento: ",  metrics.mean_squared_error(reg.predict(x_train),y_train))
    #Predecimos los valores de las etiquetas del conjunto de test
    pred=reg.predict(x_test)
    print("Error en el conjunto de test: ",  metrics.mean_squared_error(y_test,pred))    
    
    continuar()
    
    
print("AJUSTE DE LOS MODELOS")

if TUNING:

    #Regresión lineal con regularización de tipo Ridge
    reg1=Ridge(solver='auto',random_state=24,tol=0.0001,max_iter=10000)
    #svm para regresión con kernel linear  
    reg2=LinearSVR(loss='squared_epsilon_insensitive',dual=False,tol=0.0001, max_iter=10000, random_state=24)
   
    # Determinamos los valores de los parámetros a probar en el grid
    # np.logspace genera un conjunto de puntos equiespaciados según una escala logarítmica
    # El primer y segundo parámetro indican el exponente al que se eleva la base 10, para el primer y último punto resp.
    # El último parámetro indica el número de puntos intermedios a generar
    param_grid1={'alpha': np.logspace(-5,5,11)}
    param_grid2={'C': np.logspace(-5,5,11), 'epsilon':np.logspace(-6,-1,6)}
    
    #Definimos el grid para cada uno de los modelos

    grid1 = GridSearchCV(estimator=reg1,
                   param_grid=param_grid1,
                   n_jobs = -1,  #para que use todos los procesadores del ordenador
                   scoring='neg_mean_squared_error',  #métrica a usar
                   cv=5)  #cross validation con k=5
    grid2 = GridSearchCV(estimator=reg2,
                   param_grid=param_grid2,
                   n_jobs = -1,  #para que use todos los procesadores del ordenador
                   scoring='neg_mean_squared_error',  #métrica a usar
                   cv=5)  #cross validation con k=5

    # Buscamos los mejores parámetros y ajustamos el modelo con los parámetros mejores al conjunto de entrenamiento
    reg1=grid1.fit(x_train_prep,y_train)
    # Mostramos los resultados obtenidos
    printCV(x_train_prep, y_train, reg1, "--------Ridge--------", True)
    
    # Buscamos los mejores parámetros y ajustamos el modelo con los parámetros mejores al conjunto de entrenamiento
    reg2=grid2.fit(x_train_prep,y_train)
    # Mostramos los resultados obtenidos
    printCV(x_train_prep, y_train, reg2, "--------Support Vector Regression--------", True)
        
    
else:
    #Regresión lineal con regularización de tipo Ridge
    reg1=Ridge(alpha=0.01, tol=0.0001, solver='auto',random_state=24, max_iter=10000)  
    #svm para regresión con kernel linear  
    reg2=LinearSVR(epsilon=0.000001,loss='squared_epsilon_insensitive',dual=False,tol=0.0001, C=10000, max_iter=10000, random_state=24)    
    

    # Mostramos los resultados obtenidos
    printCV(x_train_prep, y_train, reg1,"--------Ridge--------", False)
    printCV(x_train_prep,y_train, reg2,"--------Support Vector Regression--------", False)

best_model=reg2
name="---------Support Vector Regression--------"
printFinalResults(x_test_prep, x_train_prep, best_model, name)


print("ENTRENAMIENTO SOBRE TODOS LOS DATOS")
#Recuperamos los datos originales
x = x_train_orig.copy()
y = y_train_orig.copy()

#Unimos todos los datos originales (entrenamiento y test) para entrenar al modelo
x=np.append(x,x_test,axis=0)
y=np.append(y,y_test)

#Permutamos las instancias, para que los datos de test no estén todos al final
x,y=shuffle(x,y,random_state=24)

#Preprocesamos el nuevo conjunto de datos
x=scaler.fit_transform(x) #Normalizamos los datos

#Eliminamos los outliers
outliers=detectOutliers(x)
x=removeExamples(x,outliers)
y=removeExamples(y,outliers)

#Nos quedamos con las características que aportan más información
#las que tienen un valor de mutual information (MI) mayor a 0.2
index=dependence(x,y,0.2)
x=removeFeatures(x,index)

print("Nuevo tamaño de la matriz de características:", x.shape)
print("Nuevo tamaño del vector de etiquetas:", len(y))

continuar()
#Mostramos el accuracy de validación cruzada obtenido
printCV(x, y, best_model,name, False) 
print(" FIN DEL PROBLEMA DE REGRESIÓN ")
