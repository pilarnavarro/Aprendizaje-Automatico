# -*- coding: utf-8 -*-
"""
PROYECTO FINAL
Pilar Navarro Ramírez y Alejandro Miguel Palencia Blanco
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from matplotlib import cm
from matplotlib.lines import Line2D
from collections import Counter
from joblib import Parallel, delayed
from math import sqrt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from RBF import *

# Fijamos la semilla
np.random.seed(1)

"""# CONSTANTES"""

RESIZE_SHAPE = (27,27)  #Tamaño resultante de una imagen tras reescalarla

VISUALIZATION = False #Determina si se visualizan los datos mediante PCA y t-SNE
TUNING = False  #Determina si se ejecuta o no el ajuste de parámetros de los distintos modelos
NPY = True      #Indica si se cargan los datos ya guardados en disco en formato de array de numpy 
DOWNSAMPLING = True #Determina si se aplica downsampling en el preprocesamiento

K_FOLDS = 5    #Número de particiones para validación cruzada
N_JOBS = -1    #Número de procesadores a usar para la ejecución en paralelo. -1 es para usar todos los procesadores

# Hiperparámetros de Regresión Logística
C = 0.07435854122563143  # Parámetro de regularización

# Hiperparámetros de MLP
ACTIVATION = 'relu'
NUM_NEURONS = 100
ALPHA = 0.14729982381618334
EARLY_STOPPING = False

# Hiperparámetros de Random Forest
NUM_ARBOLES=290 #Número de árboles para random forest
CCP_ALPHA=0 #Valor para el parámetro de complejidad para el pruning en random forest

# Hiperparámetros de Support Vector Machine
C_SVM= 100 #Parámetro de regularización para SVM
GAMMA= 0.01 #Valor del parámetro gamma para SVM

# Hiperparámetros de RBF-Network
K=1500 #Valor para el número de clusters en RBF-Network

"""# FUNCIONES"""

def continuar():
    input("\n----Presiona Enter para continuar----\n")




def loadDataset(directory_path, shuffle=True):
    """ Función para cargar las imágenes en forma de arrays de numpy

        Args:
            directory_path: ruta donde se encuentran los datos
            shuffle: booleano que determina si se baraja o no el conjunto de datos.
    """

    import glob, os
    
    current_dir = os.path.abspath(os.getcwd())
    os.chdir(directory_path)
    imgs, labels = [], []
    
    for class_dir in os.listdir('./'):
        print("Class:",class_dir)
        class_name = class_dir
        for i, file in enumerate(glob.glob("{}/*.png".format(class_dir))):
            imgs.append(cv2.imread(file, 0))
            labels.append(class_name)
        
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)

    if shuffle:
        indices = np.random.permutation(len(imgs))
        imgs = imgs[indices]
        labels = labels[indices]
            
    os.chdir(current_dir)
    
    return imgs, labels




def plot2D(x,y,title):
    """ Función para visualizar una muesta etiquetada en 2D

        Args: 
            x: muestra de puntos a visualizar
            y: vector de etiquetas asociado a la muestra x
            title: título para el gráfico generado 
        """
    
    num_classes = np.unique(y).size

    plt.figure(figsize=(10,10))
    plt.scatter(x[:,0],x[:,1],c=y, cmap='tab20', alpha=0.5)

    cmap=cm.get_cmap('tab20')

    proxys=[]
    labels=[]
    for l in range(num_classes):
        proxys.append(Line2D([0],[0], linestyle='none', c=cmap(l/(num_classes-1)), marker='o'))
        labels.append(str(l+1))

    plt.legend(proxys, labels, numpoints = 1,framealpha=0.5)
    plt.title(title)
    plt.show()




def visualization(x,y,title):
    """ Función para visualizar una muesta etiquetada en 2D, 
        tras reducir su dimensionalidad con PCA y posteriormente con T-SNE

        Args: 
            x: muestra de puntos a visualizar
            y: vector de etiquetas asociado a la muestra x
            title: título para el gráfico generado 
    """

    #Escalamos las características para que tengan media 0 y varianza 1
    x = StandardScaler().fit_transform(x)
    #Realizamos un análisis de componentes principales, para quedarnos con 2 componentes y poder visualizar los datos
    pca=PCA(n_components=2, random_state=1)
    x_pca=pca.fit_transform(x)
    #Visualizamos los datos resultantes en 2D
    plot2D(x_pca, y, 'PCA\n'+title)
    #Vemos la varianza explicada por cada una de las dos componentes
    print("Varianza explicada: ", pca.explained_variance_ratio_)
    #Reducimos ahora la dimensionalidad con t-SNE, partiendo de los resultados obtenidos con pca
    x_tsne = TSNE(n_components=2, init=x_pca,perplexity=30).fit_transform(x)
    #Visualizamos los datos resultantes en 2D
    plot2D(x_tsne, y, 'TSNE\n'+title)




def boundingBox(img):
    """ Función para calcular la caja englobante del carácter 
    representado en la imagen que se pasa como parámetro 
    
    Args:
        img: imagen en la que se quiere calcular la caja englobante
    
    Returns:
        x,y: coordenadas de la esquina superior izquierda de la caja
        w: ancho de la caja
        h: altura de la caja
    """
    #Referencias:
    #https://learnopencv.com/otsu-thresholding-with-opencv/
    #https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    #https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

    kernel = np.ones((3,3),np.uint8) #Kernel de 1's y tamaño 3x3
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) #Calculamos el closing de la imagen para rellenar huecos 
    _,thresh = cv2.threshold(closing,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Método de Otsu para calcular el threashold de la imagen automáticamente
    x,y,w,h = 2,2,thresh.shape[0]-2,thresh.shape[1]-2 #Le restamos 2 para quitarle el padding que tienen las imágenes en el dataset
    #Acotamos los límites del BB hasta encontrar un píxel blanco (que forme parte del trazo del carácter)
    while not any(thresh[y:y+h,x]>0):
        x += 1
        w -= 1
    while not any(thresh[y,x:x+w]>0):
        y += 1
        h -= 1
    while not any(thresh[y:y+h,x+w-1]>0):
        w -= 1
    while not any(thresh[y+h-1,x:x+w]>0):
        h -= 1
    return x,y,w,h



def meanDimensionsBoundingBox(data):
    """ Función para calcular el ancho y alto medio 
        de las cajas englobantes del conjunto de datos
        pasado como parámetro
        
        Args:
            data: conjunto de imágenes
        Returns:
            w_mean: ancho medio de las cajas englobantes
            h_mean : altura media de las cajas englobantes
    """

    w_mean, h_mean = 0.0, 0.0
    for i in range(data.shape[0]):
        x,y,w,h = boundingBox(data[i])
        w_mean += w
        h_mean += h
    w_mean /= data.shape[0]
    h_mean /= data.shape[0]
    return w_mean, h_mean




def preprocessImage(img):
    """ Función para preprocesar una imagen pasada como parámetro
    
    Args:
        img: imagen a preprocesar
    Returns:
        imagen preprocesada
    """
    #Referencias:
    #https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff

    x,y,w,h = boundingBox(img)  #Determinamos la caja englobante del caracter en la imagen
    img_prep = img[y:y+h,x:x+w] #Nos quedamos con los píxeles englobados por la caja ('recortamos' la imagen)
    img_prep = cv2.resize(img_prep,RESIZE_SHAPE,interpolation=cv2.INTER_LINEAR) #Reescalamos la imagen recortada, con interpolación bilineal
    #Hacemos downsampling de la imagen, para reducir el número de píxeles
    if DOWNSAMPLING:
        img_prep = cv2.GaussianBlur(img_prep,ksize=(3,3),sigmaX=0.5) #Aplicamos un filtro gaussiano para prevenir el aliasing, con kernel 
                                                                    #de tamaño 3x3, y sigma=0.5
        img_prep = img_prep[1:img_prep.shape[0]:2,1:img_prep.shape[1]:2] #Eliminamos las filas y columnas pares
    img_prep = np.reshape(img_prep,img_prep.size) #Pasamos la imagen a forma de vector
    return img_prep




def preprocess(data):
    """ Fundión para preprocesar todas las imágenes 
        del conjunto de datos pasado como parámetro en paralelo
        
        Args:
            data:conjunto de imágenes a preprocesar
        Returns: 
            Conjunto de imágenes preprocesado
    """

    out = Parallel(n_jobs=-1)(map(delayed(preprocessImage),data)) #Preprocesamos en paralelo cada una de las imágenes
    out = np.array(out,np.float32)
    return out




def plotMatrix(matrix, title, labels=False):
    """ Función para visualizar una matriz como un mapa de calor

    Args: 
        matrix: matriz a visualizar
        title: título para el gráfico generado 
        labels: booleano que determina si se añaden etiquetas a los ejes
    """
    
    plt.figure(figsize=(8,8))
    plt.matshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title,fontsize=15)
    if labels:
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
    plt.show()




def findRange(model, param_name, x_train, y_train, params, 
            color='black', log=True, n_jobs_cv=None):
    """ Función que determina un intervalo de valores para un parámetro 
        entre los cuales el modelo presenta los mejores resultados
        
        Args:
            model: modelo cuyo parámetro se quiere ajustar
            param_name: parámetro para el que se quiere buscar el intervalo
                de mejores valores
            x_train: conjunto de entrenamiento
            y_train: vector de etiquetas asociado al conjunto de entrenamiento
            params: lista de valores para el parámetros en la que se busca 
                el mejor intervalo
            color: color para los puntos en la visualización
            log: booleano que determina si se usa escala logarítmica en el eje x en la visualización
            n_jobs_cv: número de procesadores usados en paralelo en validación cruzada
        Returns:
            a, b : extremos del intervalo
            a_score, b_score: accuracy del modelo para los valores a y b del parámetro, resp.
    """
    scores = []
    
    print("Búsqueda del intervalo con mejores valores para el parámetro "+param_name)
    for i in range(len(params)):
        model.set_params(**{param_name:params[i]})
        #Aplicamos cross_validation con K_FOLDS particiones para determinar el accuracy del modelo
        # y el resultado se añade al vector de scores
        scores.append(np.mean(cross_val_score(model,x_train,y_train,cv=K_FOLDS,n_jobs=n_jobs_cv)))
        print(params[i],":",scores[i])
    
    params = np.array(params)
    scores = np.array(scores)
    
    plotScores(params, scores, color, 'Accuracy media frente a '+param_name, param_name, log=log)

    #Índice del valor del parámetro con accuracy más alta
    max_index = np.argmax(scores)
    #Si el mejor valor es el primero o el último proporcionados, se devuelve ese valor
    a = b = params[max_index]
    a_score = b_score = scores[max_index]
    if max_index>0 and max_index<scores.size-1:
        #Determinamos si el intervalo con mejores resultados queda por encima o por debajo
        #del índice con mejor valor determinado
        if scores[max_index-1] > scores[max_index+1]:
            #Establecemos el extremo del intervalo que falta y el valor de accuracy del mismo
            a = params[max_index-1]
            a_score = scores[max_index-1]
        else:
            #Establecemos el extremo del intervalo que falta y el valor de accuracy del mismo
            b = params[max_index+1]
            b_score = scores[max_index+1]
    return a, b, a_score, b_score




def dichotomicSearch(model, param_name, x_train, y_train, a, b, a_score=-1, b_score=-1, 
                    tol=0.001, integer=False, tuple_param=False, n_jobs_cv=None):
    """ Función que aplica búsqueda dicotómica sobre un modelo para estimar
        el mejor valor del hiperparámetro especificado como argumento

        Args: 
            model: modelo cuyo hiperparámetro se quiere ajustar
            param_name: nombre del hiperparámetro a ajustar
            x_train: conjunto de entrenamiento
            y_train: vector de etiquetas asociado al conjunto de entrenamiento
            a: extremo inferior del intervalo de búsqueda
            b: extremo superior del intervalo de búsqueda
            a_score: score cuando el hiperparámetro toma el valor a. 
            Será -1 si no se conoce el score. 
            b_score: score cuando el hiperparámetro toma el valor b.
            Será -1 si no se conoce el score.
            tol: tolerancia para el criterio de parada
            integer: booleano que determina si el hiperparámetro 
                es de tipo entero o no. En caso negativo se asume que es float. 
            tuple_param: booleano que determina si el hiperparámetro es una tupla
            n_jobs_cv: número de procesadores usados en paralelo en validación cruzada
        
        Return:
            params: lista de valores probados para el hiperparámetro
            score: lista de scores para los valores probados
    """
    print("Búsqueda dicotómica para "+ param_name)
    if a_score == -1:
        if tuple_param:
            model.set_params(**{param_name:(a,a)})
        else:
            model.set_params(**{param_name:a})
        #Aplicamos cross_validation con K_FOLDS particiones para determinar el accuracy del modelo
        a_score = np.mean(cross_val_score(model,x_train,y_train,cv=K_FOLDS,n_jobs=n_jobs_cv))
    print(a,":",a_score)
    if b_score == -1:
        if tuple_param:
            model.set_params(**{param_name:(b,b)})
        else:
            model.set_params(**{param_name:b})
        #Aplicamos cross_validation con K_FOLDS particiones para determinar el accuracy del modelo
        b_score = np.mean(cross_val_score(model,x_train,y_train,cv=K_FOLDS,n_jobs=n_jobs_cv))
    print(b,":",b_score)
    
    params = [a, b]
    scores = [a_score, b_score]
    
    while b-a > tol:
        if a_score < b_score:
            a = (a+b)/2
            if integer:
                a = int(a)
            if tuple_param:
                model.set_params(**{param_name:(a,a)})
            else:
                model.set_params(**{param_name:a})
            a_score = np.mean(cross_val_score(model,x_train,y_train,cv=K_FOLDS,n_jobs=n_jobs_cv))
            params.append(a)
            scores.append(a_score)
            print(a,":",a_score)
        else:
            b = (a+b)/2
            if integer:
                b = int(b)
            if tuple_param:
                model.set_params(**{param_name:(b,b)})
            else:
                model.set_params(**{param_name:b})
            b_score = np.mean(cross_val_score(model,x_train,y_train,cv=K_FOLDS,n_jobs=n_jobs_cv))
            params.append(b)
            scores.append(b_score)
            print(b,":",b_score)
    
    return np.array(params), np.array(scores)




def printFinalResults(model,model_name, x_train, y_train, x_test, y_test):
    """
    Función que imprime por pantalla los resultados en el conjunto
    de test y conjunto de entrenamiento del clasificador indicado
        
    Args: 
        model: modelo cuya calidad se quiere medir
        model_name: nombre del modelo
        x_train: conjunto de datos preprocesados, para el entrenamiento del modelo
        y_train: etiquetas asociadas al conjunto de entrenamiento
        x_test: conjunto de datos de test preprocesados
        y_test: etiquetas asociadas al conjunto de test
    """
    
    print(model_name)
    #Entrenamos el clasificador
    model.fit(x_train,y_train)
    print("Accuracy en el conjunto de entrenamiento: ", model.score(x_train,y_train))
    #Predecimos los valores de las etiquetas del conjunto de test
    pred=model.predict(x_test)
    print("Accuracy en el conjunto de test: ", metrics.accuracy_score(y_test,pred))
    continuar()
    #Determinamos la matriz de confusión
    #normalize=all hace que todos los valores de la matriz estén normalizados
    mat=metrics.confusion_matrix(y_test,pred,normalize='all')
    #Visualizamos la matriz de confusión
    plotMatrix(mat, "Matriz de confusión para los datos de test\n",True)
    continuar()

"""# LEER DATOS DE DISCO"""

if NPY:
    x_train = np.load("./datos/npy/x_train.npy")
    y_train = np.load("./datos/npy/y_train.npy")
    x_test = np.load("./datos/npy/x_test.npy")
    y_test = np.load("./datos/npy/y_test.npy")
else:
    x_train, y_train = loadDataset('./datos/DevanagariHandwrittenCharacterDataset/Train/')
    x_test, y_test = loadDataset('./datos/DevanagariHandwrittenCharacterDataset/Test/')


    np.save("datos/npy/x_train.npy", x_train)
    np.save("datos/npy/y_train.npy", y_train)
    np.save("datos/npy/x_test.npy", x_test)
    np.save("datos/npy/y_test.npy", y_test)

# Codificar etiquetas con números del 0 al 45
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print("---------------CLASIFICACIÓN DE CARACTERES DEVANAGARI----------------")
print("Tamaño del conjunto de entrenamiento: ",x_train.shape)
print("Tamaño del conjunto de test: ",x_test.shape)
#print("Número de ejemplos en cada clase de training:\n",Counter(y_train)) #Las clases están balanceadas
#print("Número de ejemplos en cada clase de test:\n",Counter(y_test))
continuar()

#Nos guardamos una copia de los datos de entrenamiento originales
x_train_orig=x_train.copy()
y_train_orig=y_train.copy()

#Convertimos cada una de las matrices representando una imagen a un vector (1d)
x_train_orig = np.reshape(x_train_orig,(x_train_orig.shape[0],x_train_orig.shape[1]*x_train_orig.shape[2]))

"""# PREPROCESAMIENTO"""

print("PREPROCESAMIENTO")

print("Visualización de un ejemplo de preprocesamiento")

img_example = x_train[107].copy()
plt.imshow(img_example, cmap='gray')
plt.show()
continuar()
kernel = np.ones((3,3),np.uint8)
img_example = cv2.morphologyEx(img_example,cv2.MORPH_CLOSE,kernel)
plt.imshow(img_example, cmap='gray')
plt.show()
continuar()
_,img_example = cv2.threshold(img_example,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
x,y,w,h = boundingBox(img_example)
plt.imshow(img_example, cmap='gray')
plt.gca().add_patch(Rectangle((x,y),w-1,h-1,linewidth=2,edgecolor='turquoise',facecolor='none'))
plt.show()
continuar()
img_example = img_example[y:y+h,x:x+w]
img_example = cv2.resize(img_example,RESIZE_SHAPE,interpolation=cv2.INTER_LINEAR)
plt.imshow(img_example, cmap='gray')
plt.show()
continuar()
img_example = cv2.GaussianBlur(img_example,ksize=(3,3),sigmaX=0.5)
img_example = img_example[0:img_example.shape[0]:2,0:img_example.shape[1]:2]
plt.imshow(img_example, cmap='gray')
plt.show()
continuar()

print("Tamaño medio de las cajas englobantes en el conjunto de entrenamiento:", meanDimensionsBoundingBox(x_train))
continuar()

#Preprocesamos todas las imágenes del conjunto de entrenamiento y test
x_train = preprocess(x_train)
x_test = preprocess(x_test)

print("Nuevo número de características tras el preprocesamiento: ", len(x_train[0]))
continuar()

#Eliminamos las características con varianza 0, es decir, que son constantes
varthresh = VarianceThreshold()
varthresh.fit(x_train)
x_train_prep = varthresh.transform(x_train) 
x_test_prep = varthresh.transform(x_test)
print("Nuevo número de características tras eliminar las de varianza nula:", len(x_train[0]))
continuar()

#Escalamos las características para que tengan media 0 y varianza 1
scaler = StandardScaler()
scaler.fit(x_train_prep) #Normalizamos los datos
x_train = scaler.transform(x_train_prep) 
x_test = scaler.transform(x_test_prep)
continuar()

"""# VISUALIZACIÓN"""

if VISUALIZATION:
    print("Visualizamos los datos antes del preprocesamiento")

    #Nos quedamos con el 50% de los datos de entrenamiento para visualizarlos
    x, _, y, _ = train_test_split(x_train_orig, y_train, stratify=y_train, train_size=0.5)

    x1=[x[i] for i in range(len(x)) if 0<=y[i]<18] #Clases a visualizar
    y1=[y[i] for i in range(len(y)) if 0<=y[i]<18]
    visualization(x1,y1,'Caracteres del 1 al 18')
    continuar()
    x2=[x[i] for i in range(len(x)) if 18<=y[i]<36] #Clases a visualizar
    y2=[y[i] for i in range(len(y)) if 18<=y[i]<36]
    visualization(x2,y2,'Caracteres del 19 al 36')
    continuar()
    x3=[x[i] for i in range(len(x)) if 36<=y[i]<46] #Clases a visualizar
    y3=[y[i] for i in range(len(y)) if 36<=y[i]<46]
    visualization(x3,y3,'Dígitos del 0 al 9')
    continuar()

    print("Visualizamos los datos tras el preprocesamiento")

    #Nos quedamos con el 50% de los datos de entrenamiento preprocesados para visualizarlos
    x, _, y, _ = train_test_split(x_train, y_train, stratify=y_train, train_size=0.5)

    x1=[x[i] for i in range(len(x)) if 0<=y[i]<18] #Clases a visualizar
    y1=[y[i] for i in range(len(y)) if 0<=y[i]<18]
    visualization(x1,y1,'Caracteres del 1 al 18')
    continuar()
    x2=[x[i] for i in range(len(x)) if 18<=y[i]<36] #Clases a visualizar
    y2=[y[i] for i in range(len(y)) if 18<=y[i]<36]
    visualization(x2,y2,'Caracteres del 19 al 36')
    continuar()
    x3=[x[i] for i in range(len(x)) if 36<=y[i]<46] #Clases a visualizar
    y3=[y[i] for i in range(len(y)) if 36<=y[i]<46]
    visualization(x3,y3,'Dígitos del 0 al 9')
    continuar()

"""# AJUSTE DE MODELOS"""

print("AJUSTE DE LOS MODELOS")


#Regresión Logística
log_reg = LogisticRegression(penalty='l2',
                            tol=0.001,
                            solver='saga',
                            max_iter=1200,
                            multi_class='multinomial',
                            n_jobs=N_JOBS,
                            random_state=1)

#Perceptrón Multicapa
mlp = MLPClassifier(hidden_layer_sizes=(100,100),
                    solver='adam',
                    max_iter=1000,
                    random_state=1)

#Random Forest
rf=RandomForestClassifier(n_jobs=N_JOBS,random_state=24)

#Support Vector Machines con Kernel RBF
svm=SVC(kernel='rbf',random_state=24,tol=0.01,max_iter=1000)

#Valor del parámetro R usado en el cálculo de r para las funciones de base raial
#Es el diámetro del conjunto de datos de entrenamiento preprocesados
#R=computeR(x_train_prep) #Tarda bastante en calcularse
R=2677.163

if TUNING:
    print("------------Regresión Logística-------------")
    #Valores del parámetro de regularización a probar
    params = np.logspace(-3,2,11)
    #Estimación de parámetro de regularización
    a, b, a_score, b_score = findRange(log_reg, 'C', x_train, y_train, params, 'blue')
    continuar()
    params, scores = dichotomicSearch(log_reg, 'C', x_train, y_train, a, b, a_score, b_score, tol=0.01)
    continuar()
    plotScores(params, scores, 'blue', 'Accuracy media frente a C', 'C')
    continuar()
    #Determinamos el índice del valor del parámetro que ofrece los mejores resultados
    max_index = np.argmax(scores)
    best_param = params[max_index] #Valor del parámetro con mejor accuracy
    best_score = scores[max_index] #Mejor accuracy del modelo obtenido
    print("Mejor configuración de hiperparámetros:")
    print("\tC =",best_param)
    print("Accuracy con validación cruzada del mejor estimador: ")
    print("\tE_cv =",best_score)
    continuar()
    
    print("------------Perceptrón Multicapa-------------")
    # Estimación del número de neuronas con la función de activación 'relu'
    mlp.set_params(**{'activation':'relu'})
    params_relu, scores_relu = dichotomicSearch(mlp, 'hidden_layer_sizes', x_train, y_train, 50, 100, 
                                                tol=1, integer=True, n_jobs_cv=N_JOBS)
    # Estimación del número de neuronas con la función de activación 'tanh'
    mlp.set_params(**{'activation':'tanh'})
    params_tanh, scores_tanh = dichotomicSearch(mlp, 'hidden_layer_sizes', x_train, y_train, 50, 100, 
                                                tol=1, integer=True, n_jobs_cv=N_JOBS)
    # Estimación del número de neuronas con la función de activación 'logistic'
    mlp.set_params(**{'activation':'logistic'})
    params_logistic, scores_logistic = dichotomicSearch(mlp, 'hidden_layer_sizes', x_train, y_train, 50, 100, 
                                                        tol=1, integer=True, n_jobs_cv=N_JOBS)
    # Gráfica que muestra los resultados obtenidos en cada una de las estimaciones
    plt.scatter(params_relu,scores_relu,c='red',label='relu')
    plt.scatter(params_tanh,scores_tanh,c='green',label='tanh')
    plt.scatter(params_logistic,scores_logistic,c='purple',label='logistic')
    plt.title('Accuracy media frente al número de neuronas en las capas ocultas')
    plt.xlabel('Número de neuronas en las capas ocultas')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
    # Selección de la configuración número de neuronas / activación que 
    #     obtiene los mejores resultados
    max_index_relu = np.argmax(scores_relu)
    max_index_tanh = np.argmax(scores_tanh)
    max_index_logistic = np.argmax(scores_logistic)
    activation_scores = [scores_relu[max_index_relu],
                        scores_tanh[max_index_tanh],
                        scores_logistic[max_index_logistic]]
    best_score = best_param_activation = best_param_num_neurons = None
    if np.argmax(activation_scores) == 0:
        best_score = scores_relu[max_index_relu]
        best_param_activation = 'relu'
        best_param_num_neurons = params_relu[max_index_relu]
    elif np.argmax(activation_scores) == 1:
        best_score = scores_tanh[max_index_tanh]
        best_param_activation = 'tanh'
        best_param_num_neurons = params_tanh[max_index_tanh]
    else:
        best_score = scores_logistic[max_index_logistic]
        best_param_activation = 'logistic'
        best_param_num_neurons = params_tanh[max_index_logistic]
    print(best_param_num_neurons,",",best_param_activation,":",best_score)
    mlp.set_params(**{'hidden_layer_sizes':(best_param_num_neurons,best_param_num_neurons)})
    mlp.set_params(**{'activation':best_param_activation})
    #Valores del parámetro de regularización a probar
    params_alpha = np.logspace(-5,1,13)
    # Estimación del parámetro de regularización alpha con valores en distintas escalas
    a, b, a_score, b_score = findRange(mlp, 'alpha', x_train, y_train, params_alpha, 'black', log=True, n_jobs_cv=N_JOBS)
    params_alpha, scores = dichotomicSearch(mlp, 'alpha', x_train, y_train, a, b, a_score, b_score, n_jobs_cv=N_JOBS)
    # Gráfica que muestra los resultados obtenidos en la búsqueda dicotómica
    plotScores(params_alpha, scores, 'black', 'Accuracy media frente a alpha', 'alpha')
    # Selección del mejor valor para el parámetro alpha
    max_index_alpha = np.argmax(scores)
    best_param_alpha = params_alpha[max_index_alpha]
    best_score = scores[max_index_alpha]
    mlp.set_params(**{'alpha':best_param_alpha})
    print(best_param_alpha,":",best_score)
    # Estimación del parámetro early_stopping
    mlp.set_params(**{'early_stopping':True})
    score_early_stopping = np.mean(cross_val_score(mlp,x_train,y_train,cv=K_FOLDS,n_jobs=N_JOBS))
    print("early_stopping= True :",score_early_stopping)
    # Selección del mejor valor para el parámetro early_stopping
    best_param_early_stopping = False
    if score_early_stopping > best_score:
        best_score = score_early_stopping
        best_param_early_stopping = True
    else:
        mlp.set_params(**{'early_stopping':False})
    print("early_stopping=",best_param_early_stopping,":",best_score)
    print("Mejor configuración de hiperparámetros:")
    print("\tnum_neurons =",best_param_num_neurons)
    print("\tactivation =",best_param_activation)
    print("\talpha =",best_param_alpha)
    print("\tearly_stopping =",best_param_early_stopping)
    print("Estimación de E_out a partir del error en validación cruzada:")
    print("\tE_out =",best_score)
    continuar()

    print("--------------Random Forest-----------------")
    #Estimación del número de árboles
    params,scores=dichotomicSearch(rf, "n_estimators" , x_train,y_train,100,300,-1,-1,1,True,False,N_JOBS)
    continuar()
    #Visualizamos los valores de accuracy obtenidos para los distintos valores de los parámetros probados, con y sin límites para los ejes
    plotScores(params,scores,'orange','Accuracy media para el número de árboles','Número de árboles', log=False)
    plotScores(params,scores,'orange','Accuracy media para el número de árboles','Número de árboles', log=False,xlimits=[240,300],ylimits=[0.92,0.923])
    continuar()
    #Determinamos el índice del valor del parámetro que ofrece los mejores resultados
    max_index = np.argmax(scores)
    best_param = params[max_index] #Valor del parámetro con mejor accuracy
    rf.set_params(**{'n_estimators':best_param})
    #Estimación del parámetro de complejidad para el pruning    
    a,b,a_score, b_score=findRange(rf,"ccp_alpha", x_train, y_train, np.logspace(-6,-1,6),'orange')
    rf.set_params(**{'ccp_alpha':a})
    print("Mejor configuración de hiperparámetros:")
    print("\tNúmero de árboles =",best_param,"\tAlpha = ", a)
    print("Accuracy con validación cruzada del mejor estimador: ",)
    print("\tE_cv =",a_score)
    continuar()

    print("------Support Vector Machines con Kernel Gaussiano RBF-------")
    #Estimación del parámetro de regularización
    a,b,a_score, b_score=findRange(svm,"C", x_train, y_train, np.logspace(-2,2,5),'turquoise')
    continuar()
    best_param1 = b #Valor del parámetro con mejor accuracy
    svm.set_params(**{'C':best_param1})
    continuar()
    
    #Estimación del valor de gamma
    a,b,a_score, b_score=findRange(svm,"gamma", x_train, y_train, np.logspace(-3,1,5),'turquoise')
    continuar()
    best_param2 = b #Valor del parámetro con mejor accuracy
    best_score = b_score #Mejor accuracy del modelo obtenido
    print("Mejor configuración de hiperparámetros:")
    print("\tC = ",best_param1,"\tgamma = ",best_param2 )
    print("Accuracy con validación cruzada del mejor estimador: ")
    print("\tE_cv =",best_score)
    continuar()

    print("----------Red de funciones de base radial------------")
    #Estimación del mejor valor de K
    params=[100,500,1000,1500]
    scores=findBestK(x_train_prep, y_train, params, R)
    continuar()
    max_index = np.argmax(scores)
    best_param = params[max_index] #Valor del parámetro con mejor accuracy
    best_score = scores[max_index] #Mejor accuracy del modelo obtenido
    print("Mejor configuración de hiperparámetros:")
    print("\tNúmero de clusters = ",best_param)
    print("Accuracy con validación cruzada del mejor estimador: ")
    print("\tE_cv =",best_score)
    continuar()

else:
    print("\nVALIDACIÓN")
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, train_size=0.8, 
                                                        stratify=y_train, random_state=42)
    
    print("------------Regresión Logística-------------")
    log_reg.set_params(**{'C':C})
    print("Mejor configuración de hiperparámetros:")
    print("\tC =",C)
    print("Accuracy:")
    log_reg.fit(x_train2, y_train2)
    print("\tE_in =",log_reg.score(x_train2, y_train2))
    print("\tE_val =",log_reg.score(x_val, y_val))
    
    print("------------Perceptrón Multicapa-------------")
    mlp.set_params(**{'activation':ACTIVATION})
    mlp.set_params(**{'hidden_layer_sizes':(NUM_NEURONS,NUM_NEURONS)})
    mlp.set_params(**{'alpha':ALPHA})
    mlp.set_params(**{'early_stopping':EARLY_STOPPING})
    print("Mejor configuración de hiperparámetros:")
    print("\tactivation =",ACTIVATION)
    print("\tnum_neurons =",NUM_NEURONS)
    print("\talpha =",ALPHA)
    print("\tearly_stopping =",EARLY_STOPPING)
    print("Accuracy:")
    mlp.fit(x_train2, y_train2)
    print("\tE_in =",mlp.score(x_train2, y_train2))
    print("\tE_val =",mlp.score(x_val, y_val))
    
    print("---------------Random Forest------------")
    rf.set_params(**{'n_estimators':NUM_ARBOLES,'ccp_alpha':CCP_ALPHA})
    print("Mejor configuración de hiperparámetros:")
    print("\tNúmero de árboles = ",NUM_ARBOLES)
    print("\tAlpha = ",CCP_ALPHA)
    print("Accuracy:")
    rf.fit(x_train2, y_train2)
    print("\tE_in =",rf.score(x_train2, y_train2))
    print("\tE_val =",rf.score(x_val, y_val))
    
    print("---------------Support Vector Machine con Kernel Gaussiano RBF------------")
    svm.set_params(**{'C':C_SVM,'gamma':GAMMA})
    print("Mejor configuración de hiperparámetros:")
    print("\tC = ",C_SVM)
    print("\tGamma = ",GAMMA)
    print("Accuracy:")
    svm.fit(x_train2, y_train2)
    print("\tE_in =",svm.score(x_train2, y_train2))
    print("\tE_val =",svm.score(x_val, y_val))

    print("----------Red de funciones de base radial------------")
    x_train2, x_val, y_train2, y_val = train_test_split(x_train_prep, y_train, train_size=0.8, 
                                                        stratify=y_train, random_state=42)
    print("Mejor configuración de hiperparámetros:")
    print("\tNúmero de clusters = ",K)
    print("Accuracy:")
    rbf=RBF(x_train2, y_train2, x_train2, y_train2, num_of_classes=46,k=K,R=R)
    print("\tE_in =",rbf.fit())
    rbf=RBF(x_train2, y_train2, x_val, y_val, num_of_classes=46,k=K,R=R)
    print("\tE_val =",rbf.fit())

"""# ENTRENAMIENTO DE LOS MODELOS  FINALES Y ESTIMACIÓN SOBRE TEST"""

#Mostramos los resultados sobre el conjunto de test de cada uno de los modelos
print("\nRESULTADOS SOBRE EL CONJUNTO DE TEST")

printFinalResults(log_reg,"------------Regresión Logística-------------",x_train,y_train,x_test,y_test)
printFinalResults(mlp,"------------Perceptrón Multicapa-------------",x_train,y_train,x_test,y_test)
printFinalResults(rf,"------------Random Forest-------------",x_train,y_train,x_test,y_test)
printFinalResults(svm,"------Support Vector Machines con Kernel Gaussiano RBF-------",x_train,y_train,x_test,y_test)

print("----------Red de funciones de base radial------------")
rbf=RBF(x_train_prep, y_train, x_train_prep, y_train, num_of_classes=46,k=K,R=R)
print("Accuracy en el conjunto de entrenamiento: ", rbf.fit())
rbf=RBF(x_train_prep, y_train, x_test_prep,y_test, num_of_classes=46,k=K,R=R,matrix=True)
m,acc=rbf.fit()
print("Accuracy en el conjunto de test: ", acc)
continuar()
plotMatrix(m, "Matriz de confusión para los datos de test\n",True)
continuar()