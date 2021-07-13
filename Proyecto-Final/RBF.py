## Implementación de RBF-Network
## Código modificado de https://towardsdatascience.com/most-effective-way-to-implement-radial-basis-function-neural-network-fo -classification-problem-33c467803319

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

K_FOLDS=3

def computeR(x_train):
    """ Función que calcula el valor de R (diámetro) del 
        conjunto de datos de entrenamiento, x_train """
    
    dist = []
    x_train_roll = x_train.copy()
    for i in range(x_train.shape[0]):
        x_train_roll = np.roll(x_train_roll,1,axis=0)
        dist.append(np.amax(np.linalg.norm(x_train-x_train_roll,axis=1)))
    R = np.amax(dist)

    return R
        
#Clase que implementa el modelo de red de funciones de base radial 
class RBF:

    def __init__(self, X, y, tX, ty, num_of_classes, k, R, matrix=False):

        self.X = X #Conjunto de datos de entrenamiento
        self.y = y #Etiquetas asociadas al conjunto de datos de entrenamiento

        self.tX = tX #Conjunto de datos de test
        self.ty = ty #Etiquetas asociadas al conjunto de datos de test

        self.number_of_classes = num_of_classes #Número de clases diferentes en el conjunto de datos 
        self.k = k #Número de clusters (centroides) 
        self.R = R #Diámetro del conjunto de datos
        
        self.matrix=matrix #Booleano que determina si se calcula la matriz de confusión al hacer .fit

    def convert_to_one_hot(self, x, num_of_classes):
        """ Pasa las etiquetas categóricas del vector x
            a vectores one hot """
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def rbf(self, x, c, r):
        """ Calcula el valor de la función de base radial 
            gaussiana en un punto x, con respecto al centroide
            dado por c y con parámetro de escala r """
        
        distance = np.linalg.norm(x-c, ord=2)
        return np.exp(-0.5 * distance/r**2 )

    def rbf_list(self, X, centroids, coef):
        """ Método que calcula la matriz Z para el modelo lineal a ajustar,
            aplicando la función de base radial a cada punto del cojunto de datos
            X, con cada uno de los centroides en 'centroids' y parámetros de escala
            dados en 'coef' """ 
        
        RBF_list = []
        for x in X:
            RBF_list.append([self.rbf(x, c, r) for (c, r) in zip(centroids, coef)])
        RBF_list=np.insert(RBF_list,0,np.ones(X.shape[0]),axis=1) #Añade un 1 para el sesgo, al principio de cada fila
        return np.array(RBF_list)


    def fit(self):
        """ Método que ajusta una red de funciones de base radial gaussianas al conjunto de datos 
        de entrenamiento, cacula las predicciones sobre el conjunto de test y devuelve el valor de accuracy
        y matriz de confusión (si así se especifica en el atributo self.matrix) obtenidos por dicho modelo ajustado """
        
        #Cálculo de los centroides con el algoritmo de K-means
        self.centroids = KMeans(n_clusters=self.k, random_state=24, algorithm='full').fit(self.X).cluster_centers_
       
        #Cálculo de los parámetros de escala, r
        self.coef = np.repeat(self.R / np.power(self.k,1/self.X.shape[1]), self.k)
        #Calculo de la matriz Z para el ajuste lineal 
        RBF_X = self.rbf_list(self.X, self.centroids, self.coef)
        
        #Cálculo de los pesos w con el algoritmo de la pseudoinversa
        self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)
        
        #Cálculo de las predicciones del conjunto de test
        RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.coef)
        self.pred_ty = RBF_list_tst @ self.w
        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

        #Accuracy obtenida sobre el conjunto de test
        acc=accuracy_score(self.ty, self.pred_ty)
        
        if self.matrix:
            #Cálculo de la matriz de confusión
            m=confusion_matrix(self.ty, self.pred_ty, normalize='all')
            return m,acc
        
        return acc
        
def cross_val_rbf(x, y, folds, k, R):
    """ Función que implementa validación cruzada 
        para una red de funciones de base radial 
             
        Args:
            x: conjunto de datos sobre el que aplicar validación cruzada
            y: vector de etiquetas asociado al conjunto de datos
            folds: número de particiones para validación cruzada
            k: número de clusters para la RBF-Network
            R: diámetro del conjunto de datos
       Returns:
           Accuracy media obtenida por el modelo RBF-Net
           en las 'folds' particiones con validación cruzada        
    """
    
    #Hago validación cruzada con 'folds' conjuntos diferentes 
    l = len(x)
    l = int(l/folds)

    datos = []               #Vector de datos para training de cada iteración
    im_datos = []            #Vector de etiquetas para training de cada iteración
    datos_val = []           #Vector de datos para validación de cada iteración
    im_datos_val = []        #Vector de etiquetas para validación de cada iteración

    for i in range(folds):
            aux_x = np.concatenate((x[:i*l], x[(i+1)*l:]), axis=0)
            aux_y = np.concatenate((y[:i*l], y[(i+1)*l:]), axis=0)
            datos.append(aux_x) #Valores de entrenamiento
            im_datos.append(aux_y)
            datos_val.append(x[i*l:(i+1)*l]) #Valores de validación
            im_datos_val.append(y[i*l:(i+1)*l])

    datos = np.array(datos)
    im_datos = np.array(im_datos)
    datos_val = np.array(datos_val)
    im_datos_val = np.array(im_datos_val)

    acc = 0
    #Ajusto el modelo
    for i in range(folds):
            rbf = RBF(datos[i], im_datos[i], datos_val[i],im_datos_val[i], num_of_classes=46,k=k,R=R)
            acc +=rbf.fit() #Accuracy acumulada 

    return  acc/folds
    
def plotScores(params,scores,color,title,xlabel, log=False,xlimits=None,ylimits=None):
    """ Función para visualizar las medidas de accuracy
        obtenidas por distintos valores de un parámetro concreto

        Args:
                params: array de numpy con los valores del parámetro
                scores: array de numpy con los valores de accuracy obtenidos para cada
                valor del parámetro que se encuentra en params
                color: color para los puntos que se visualizarán
                title: título del gráfico generado
                xlabel: etiqueta para el eje x
                log: booleano que determina si se usa escala logarítmica en el eje x
                xlimits: límites para el eje x. Será None si no se
                quiere etablecer los límites manualmente 
                ylimits: límites para el eje y. Será None si no se
                quiere etablecer los límites manualmente         
    """

    plt.scatter(params,scores, c=color)
    if xlimits!=None:
        plt.xlim(xlimits)
    if ylimits!=None:
        plt.ylim(ylimits)
    if log:
        plt.xscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


def findBestK(x_train, y_train, params, R):
    """ Función que calcula los valores de accuracy media obtenidos con validación
        cruzada para los valores del parámetro K que se indican como parámetro, y 
        visualiza dichos valores en un gráfico
        
        Args:
            x_train: conjunto de entrenamiento
            y_train: vector de etiquetas asociado al conjunto de entrenamiento
            params: lista de valores para el parámetro K a probar
            R: diámetro del conjunto de datos de entrenamiento
        Returns:
            scores: valores de accuracy en validación cruzada 
                para cada valor de K proporcionado
    """
    scores = []
    
    print("Búsqueda del mejor valor para el parámetro K")
    for i in range(len(params)):
        #Aplicamos cross_validation con 3 particiones para determinar el accuracy del modelo
        #y el resultado se añade al vector de scores
        scores.append(cross_val_rbf(x_train, y_train, K_FOLDS, params[i], R=R))
        print(params[i],":",scores[i])
    
    params = np.array(params)
    scores = np.array(scores)
    
    plotScores(params, scores, 'lightcoral', 'Accuracy media frente a K', 'Número de clusters K', log=True)
    
    return scores
