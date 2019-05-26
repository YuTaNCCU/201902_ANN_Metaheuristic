#!/usr/bin/env python
# coding: utf-8

# # 

# In[1]:
#待續
# ES 自動收斂試試看s



#%% Load Package


#https://github.com/YuTaNCCU/201902_ANN_Metaheuristic/tree/master/ES
import random
import pandas as pd
from string import ascii_lowercase
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential 
from keras.callbacks import TensorBoard,EarlyStopping
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
import  seaborn as sns
import matplotlib.pyplot as plt


#%%Load Data
url = 'https://raw.githubusercontent.com/YuTaNCCU/201902_ANN_Metaheuristic/master/Data/red.csv'
red = pd.read_csv(url)

url = 'https://raw.githubusercontent.com/YuTaNCCU/201902_ANN_Metaheuristic/master/Data/white.csv'
white = pd.read_csv(url)

red['WineCatg']='red'
white['WineCatg']='white'
Wine_Data = pd.concat([red, white])

display(
    red.shape,
  white.shape,
  Wine_Data.shape,
  Wine_Data.head(5),
  Wine_Data.tail(5)
)


Wine_Data.columns


from sklearn import preprocessing
Wine_Data_preprocessed = Wine_Data.drop(['WineCatg'], axis=1)
scaler = preprocessing.StandardScaler() 
col = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
Wine_Data_preprocessed[col] = scaler.fit_transform(Wine_Data_preprocessed[col])
Wine_Data_preprocessed.head()



print( '原本各種quality記數: \n', Wine_Data.quality.value_counts().sort_index() )
Wine_Data_Y01 = Wine_Data_preprocessed.replace({'quality':[3,4,5,6,7,8,9]},{'quality':[0,0,0,1,1,1,1]})
print( '分類成好壞兩種quality記數: \n', Wine_Data_Y01.quality.value_counts().sort_index() )
Wine_Data_Y01.head(5)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X=Wine_Data_Y01.drop(['quality'], axis=1)
y=Wine_Data_Y01['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state= 123)

display(
      X_train.shape,
      X_val.shape,
      X_test.shape,
      y_train.shape,
      y_val.shape,
      y_test.shape,
)


#%% Define Performance


def Performance(model):
    Loss, Acc = model.test_on_batch(X_train, y_train)
    print('train Loss, Acc:', Loss, Acc)
    #######################
    ## Confusion Matrix  ##
    #######################

    # Predicting the Test set results
    y_score = model.predict(X_test) #X_train X_test
    y_pred = (y_score > 0.5)  #y_pred 有 NA

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred) #y_train y_test
    #######################
    ###    accuracy      ##
    #######################
    print("Our accuracy is {}%".format(round(((cm[0][0] + cm[1][1])/sum(sum(cm)))*100,2)))

    sns.heatmap(cm,annot=True)
    #######################
    ###       AUC        ##
    #######################    
    from sklearn.metrics import roc_auc_score
    print("Our AUC is {}%".format(round((roc_auc_score(y_test, y_score)*100),2)))
    
    #######################
    ###    ROC curve     ##
    #######################
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes =2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show() 


#%% Define Keras Model


def SummarizeHistory(modelcallbacks, prfix='', UseValid=False) :
    
    plt.figure()
    plt.plot(modelcallbacks.history[prfix+'loss'])  #loss mean_absolute_error 
    if UseValid==True:
        plt.plot(modelcallbacks.history['val_'+prfix+'loss'])  #val_loss 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(modelcallbacks.history[prfix+'acc'])  #loss mean_absolute_error 
    if UseValid==True:
        plt.plot(modelcallbacks.history['val_'+prfix+'acc'])  # val_mean_absolute_error
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.show()
    
def ModelCompile():
    from keras import backend as K
    from keras.layers import Dense
    from keras.models import Sequential, Model as keras_models_Model

    K.clear_session() 
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)

    """model = Sequential()
    model.add(Dense(round(X_train.shape[1]/2), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(round(X_train.shape[1]/2), activation='relu'))
    model.add(Dense(round(X_train.shape[1]/4), activation='relu', name = 'IntermediateLayer'))
    model.add(Dense(1, activation='sigmoid'))"""
    
    model = Sequential()
    model.add(Dense(2, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def ModelFit(model, weights, epochs):
    #class_weight = {0: 1., 1: 1525/2632}
    model.set_weights(weights)
    modelcallbacks = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1,
        validation_data = (X_val, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=2, restore_best_weights=True)],
        shuffle=True) #, class_weight=class_weight
    SummarizeHistory(modelcallbacks, UseValid=True)                                        
    weights = model.get_weights() 
    return model, weights


#%% Main

#%% 只用BP
model_BP = ModelCompile()
weights = model_BP.get_weights() 
model_BP, weights = ModelFit(model=model_BP, weights=weights, epochs=5)
Performance(model_BP)
#%% Define ES Class 
class ES:
    """
    Conducts tabu search
    """
    __metaclass__ = ABCMeta

    #default hyper parameters
    InitialSigma = None
    ParentsSize = None
    ChildSize = None
    tao = None
    
    #for input/output
    KerasModels = None
    WeightsStrucure = None   
    weights = None
    
    #for record
    cur_steps = 1
    best_weight = None
    best_score = None
    best_Acc = None
    
    UseOLSReg=None
    X_train=None
    y_train=None
    
    def __init__(self, KerasModels, X_train, y_train, UseOLSReg=False, InitialSigma = 0.1, ParentsSize = 15, ChildSize = 100, tao = 0.5):
        """
        :param KerasModels: a Keras model, like keras.engine.sequential.Sequential
        :param weights: initial weights, should be a Keras model weight
        :param max_steps: maximum number of steps to run algorithm for
        :param UseOLSReg: If True, than use "OLS Regression" for the last layer
        
        """
        self.KerasModels = KerasModels
        
        self.UseOLSReg = UseOLSReg
        
        self.X_train=X_train
        self.y_train=y_train
 
        if all(isinstance(x, float) for x in [InitialSigma, tao]) and all(x > 0 for x in [InitialSigma, tao]):
            self.InitialSigma = InitialSigma
            self.tao = tao
        else:
            raise TypeError('InitialSigma & tao must be a positive float')
            
        if all(isinstance(x, int) for x in [ParentsSize, ChildSize]) and all(x > 0 for x in [ParentsSize, ChildSize]):
            self.ParentsSize = ParentsSize
            self.ChildSize = ChildSize
        else:
            raise TypeError('ParentsSize, ChildSize & max_steps must be a positive integer')

    def __str__(self): 
        return ('ES STEPS: %d ' +
                ' - BEST Accuracy: %.4f '+
                ' - BEST Score(Log Loss): %.4f ') % \
               (self.cur_steps, self.best_Acc, self.best_score)

    def __repr__(self):
        return self.__str__() 
    
    def _FlattenWeights(self, weights):
        """
        flatten weights
        
        param weights: keras神經網路的權重格式:nparray包在list中
        return WeightsStrucure : 神經網路各層的權重shape包在list中，unflatten時會用到
        return FlattenedWeights : 一維list包含所有的權重
        """
        WeightsStrucure = []
        FlattenedWeights = []
        for i_layer in weights:
            WeightsStrucure.append(i_layer.shape)
            if len(i_layer.shape) == 1 :# 該層權重的shape為一維 e.g. (15,)      
                FlattenedWeights.extend(i_layer)
            else :# 該層權重的shape為二維 e.g. (30, 15)  
                for i_links in i_layer:
                    FlattenedWeights.extend(i_links)
        return WeightsStrucure, FlattenedWeights

    def _UnflattenWeights(self, WeightsStrucure, ModifiedWeights):
        """
        Unflatten(回復成原本的結構) weights  
        
        param WeightsStrucure : 神經網路各層的權重shape包在list中
        param ModifiedWeights : 一維list包含所有meteHeuristic修改過的權重
        return: keras神經網路的權重格式:nparray包在list中
        """
        UnflattenWeights = []
        i_index = 0 
        for i_layer in WeightsStrucure:
            if len(i_layer) == 1 : # 該層權重的shape為一維 e.g. (15,)      
                TempList = ModifiedWeights[i_index:(i_index + i_layer[0])]
                TempList = np.asarray(TempList)
                i_index = i_index + i_layer[0]
            else : # 該層權重的shape為二維 e.g. (30, 15)  
                TempList = ModifiedWeights[i_index:(i_index + (i_layer[0]*i_layer[1]))]
                TempList = np.reshape(TempList, i_layer )
                i_index = i_index + (i_layer[0]*i_layer[1])
            UnflattenWeights.append(TempList)
        return UnflattenWeights   
    
    def _best(self, Population_Child_score):
        """
        Finds the best member of a neighborhood
        :param Population_Child_score: a np array
        :return: the indtex of N best member, N = ParentsSize
        """
        return np.array( Population_Child_score ).argsort()#[::-1]
    
    def _Recombination(self, Population_Parents_Weights, Population_Parents_Sigma, rows): #GenerateParents
        """
        Generate New Parents Polulation
        """
        Population_Weights_Recombination = np.zeros(shape = (rows, Population_Parents_Weights.shape[1]))
        Population_Sigma_Recombination = np.zeros(shape = (rows, Population_Parents_Weights.shape[1]))
        for index_row, _ in enumerate( Population_Weights_Recombination ):
            """
            可能可以平行計算
            """
            TwoRowschoiced = np.random.choice(Population_Parents_Weights.shape[0], size=2, replace=False,)
            Parent1Mask = np.random.randint(2, size=Population_Parents_Weights.shape[1])
            Parent2Mask = np.full(shape = Population_Parents_Weights.shape[1], fill_value = 1 )  - Parent1Mask
            
            Population_Weights_Recombination[index_row,:] = (Population_Parents_Weights[TwoRowschoiced] * [Parent1Mask, Parent2Mask]).sum(axis=0)
            Population_Sigma_Recombination[index_row,:] = Population_Parents_Sigma[TwoRowschoiced].mean(axis=0)
        return Population_Weights_Recombination, Population_Sigma_Recombination

    def _score(self, ModifiedWeights):
        
        """
        Returns objective function value of a state

        :param state: a state
        :return: objective function value of state
        """
        UnflattenedWeights = self._UnflattenWeights(WeightsStrucure = self.WeightsStrucure, ModifiedWeights = ModifiedWeights)
        self.KerasModels.set_weights(UnflattenedWeights)
        test_on_batch = self.KerasModels.test_on_batch(self.X_train, self.y_train) # return ['loss', 'acc']
        #return test_on_batch[0], test_on_batch[1] #0:loss, 1:acc
    
        return  test_on_batch[1], test_on_batch[0]
    
    def _OLSReg(self, ModifiedWeights):
        
        """
        :param : 
        :return: Keras Models, objective function value of state
        """
        UnflattenedWeights = self._UnflattenWeights(WeightsStrucure = self.WeightsStrucure, ModifiedWeights = ModifiedWeights)
        
        #OLS Regression
        #obtain the output of an intermediate layer
        #https://keras.io/getting-started/faq/?fbclid=IwAR3Zv35V-vmEy85anudOrlxCExXYwyG6cRL1UR0AaLPU6sZEoBjsbX-8LXQ#how-can-i-obtain-the-output-of-an-intermediate-layer
        self.KerasModels.set_weights(UnflattenedWeights)
        layer_name = 'IntermediateLayer'
        intermediate_layer_model = keras_models_Model(inputs=self.KerasModels.input,
                                         outputs=self.KerasModels.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(self.X_train)

        #fit LM
        lm =  LogisticRegression(random_state=0, solver='liblinear').fit(intermediate_output, self.y_train)
        
        #lm =  LinearRegression().fit(intermediate_output, self.y_train)
        # 印出係數, 截距 print(lm.coef_, lm.intercept_)
        
        #score
        #score = log_loss(y_pred = lm.predict(intermediate_output), y_true= self.y_train)
        
        #get OutLayerWeights
        OutLayerWeights = [np.array(lm.coef_).reshape(self.WeightsStrucure[-2]),
                           np.array(lm.intercept_).reshape(self.WeightsStrucure[-1])]

        #update ES-optimized weights
        UnflattenedWeights[-2:] = OutLayerWeights        
        
        #self.KerasModels.set_weights(UnflattenedWeights)
        #test_on_batch = self.KerasModels.test_on_batch(self.X_train, self.y_train, sample_weight=None) # return ['loss', 'acc']
        
        #print( 'score',score, 'test_on_batch',test_on_batch)
        _, OLS_Optimized_Weight = self._FlattenWeights(UnflattenedWeights)
        return OLS_Optimized_Weight 

    def run(self, weights, max_steps=5, verbose=10, useOLSReg = False, Population_Parents_Weights=None, Population_Parents_Sigma=None):
        """
        Conducts ES
        :param weights: 
        :param Population_Parents_Weights: 如果想要重複使用Parents Population，則由此傳入Weights
        :param Population_Parents_Sigma: 如果想要重複使用Parents Population，則由此傳入Sigma
        :param max_steps: 
        :param verbose: int which indicates how many iter to show score
        :return: Keras Models, best state and objective function value of best state
        """
            
        self.max_steps = max_steps
        
        #Step1 initial
        if (Population_Parents_Weights is None) & (Population_Parents_Sigma is None): ##接續上一個phase最好的個體
            
            if isinstance(weights, list)  :          
                self.WeightsStrucure, self.weights = self._FlattenWeights(weights)
                self.best_weight = self.weights
                self.best_Acc, self.best_score = self._score(self.best_weight)
            else:
                raise TypeError('initial_state must be a list') 
            
            Population_Parents_Weights = np.array([self.weights, self.weights])         
            Population_Parents_Sigma = np.full(shape = (self.ParentsSize, len(self.weights)), fill_value = self.InitialSigma ) 
            Population_Parents_Weights, _ = self._Recombination(Population_Parents_Weights, Population_Parents_Sigma, rows = self.ParentsSize )
            
        
        else: #接續上一個phase的10%最好的個體
            Population_Parents_Weights = Population_Parents_Weights
            Population_Parents_Sigma = Population_Parents_Sigma
            
        self.cur_steps = 1
        while True:   
            #Step2 Child
            ##Discrete Recombination
            Population_Child_Weights, Population_Child_Sigma = self._Recombination(Population_Parents_Weights, Population_Parents_Sigma, rows = self.ChildSize )
            ##mutation1
            RamdonNormalValue = np.random.normal(0, 1, 1)
            RamdonNormalValueDifferent = np.random.normal(0, 1, Population_Child_Sigma.shape)
            Population_Child_Sigma = np.exp( (1-self.tao)*RamdonNormalValue + self.tao*RamdonNormalValueDifferent )
            ##mutation2
            Population_Child_Weights = Population_Child_Weights + np.random.normal(0, Population_Child_Sigma, Population_Child_Sigma.shape)
            
            
            # OLS Regression
            if useOLSReg == True:
              for i, i_Child in enumerate(Population_Child_Weights) :
                  OLS_Optimized_Weight = self._OLSReg(i_Child)
                  #print(OLS_Optimized_Weight,'i:\n', i, Population_Child_Weights[i])
                  Population_Child_Weights[i] = OLS_Optimized_Weight
            
            
            #step3 Evaluation
            Population_Child_score = []
            for i_Child in Population_Child_Weights :
                """
                可能可以平行計算
                """
                _, tempScore = self._score(i_Child)
                Population_Child_score.append( tempScore )
            
            #選出最好的ParentsSize個個體做為下一代的親代
            BestIndex = self._best(Population_Child_score)
            BestNIndex = BestIndex[:self.ParentsSize] 
            Population_Parents_Weights = Population_Child_Weights[BestNIndex,:]
            Population_Parents_Sigma = Population_Child_Sigma[BestNIndex,:]
            #選出最好的10%個體做為下一phase的親代
            Best_10Percent_Index = BestIndex[:round(self.ChildSize*0.1)] #Multiphase
            Population_Best_10PercentChild_Weights = Population_Child_Weights[Best_10Percent_Index,:]
            Population_Best_10PercentChild_Sigma = Population_Child_Sigma[Best_10Percent_Index,:] 
            
            #更新best
            best_weight_This_Iter =  Population_Child_Weights[BestNIndex,:][0]
            _, best_score_This_Iter = self._score(Population_Child_Weights[BestNIndex,:][0])
            if best_score_This_Iter < self.best_score:
                self.best_weight =  Population_Child_Weights[BestNIndex,:][0]
                self.best_Acc, self.best_score = self._score(Population_Child_Weights[BestNIndex,:][0])
                self.cur_steps = 0#有改善就重新計步
        
            #print process 
            if ((self.cur_steps ) % verbose == 0) and verbose:
               print(self)
                
            self.cur_steps = self.cur_steps + 1
            #step4 check stop criteria
            if self.cur_steps > max_steps:
                print( 'Stop: Reach max_steps' )
                break
        return self._UnflattenWeights(WeightsStrucure = self.WeightsStrucure, ModifiedWeights = self.best_weight), self.best_score, Population_Best_10PercentChild_Weights, Population_Best_10PercentChild_Sigma

#%% Define ES Func
        
#default hyper parameters
self_InitialSigma = None
self_ParentsSize = None
self_ChildSize = None
self_tao = None

#for input/output
self_KerasModels = None
self_WeightsStrucure = None   
self_weights = None

#for record
self_cur_steps = 1
self_best_weight = None
self_best_score = None
self_best_Acc = None

self_UseOLSReg=None
self_X_train=None
self_y_train=None

def self___init__(KerasModels, X_train, y_train, UseOLSReg=False, InitialSigma = 0.1, ParentsSize = 15, ChildSize = 100, tao = 0.5):
    """
    :param KerasModels: a Keras model, like keras.engine.sequential.Sequential
    :param weights: initial weights, should be a Keras model weight
    :param max_steps: maximum number of steps to run algorithm for
    :param UseOLSReg: If True, than use "OLS Regression" for the last layer
    
    """
    global self_KerasModels, self_UseOLSReg, self_X_train, self_y_train, self_InitialSigma, self_tao, self_ParentsSize, self_ChildSize
    self_KerasModels = KerasModels
    
    self_UseOLSReg = UseOLSReg
    
    self_X_train=X_train
    self_y_train=y_train
 
    if all(isinstance(x, float) for x in [InitialSigma, tao]) and all(x > 0 for x in [InitialSigma, tao]):
        self_InitialSigma = InitialSigma
        self_tao = tao
    else:
        raise TypeError('InitialSigma & tao must be a positive float')
        
    if all(isinstance(x, int) for x in [ParentsSize, ChildSize]) and all(x > 0 for x in [ParentsSize, ChildSize]):
        self_ParentsSize = ParentsSize
        self_ChildSize = ChildSize
    else:
        raise TypeError('ParentsSize, ChildSize & max_steps must be a positive integer')

def self___str__(): 
    global self_cur_steps, self_best_Acc, self_best_score
    print ('ES STEPS: ', self_cur_steps,
            ' - BEST Accuracy: ', round(self_best_Acc,4),
            ' - BEST Score(Log Loss): ', round(self_best_score,4))

def self__FlattenWeights(weights):
    """
    flatten weights
    
    param weights: keras神經網路的權重格式:nparray包在list中
    return WeightsStrucure : 神經網路各層的權重shape包在list中，unflatten時會用到
    return FlattenedWeights : 一維list包含所有的權重
    """
    WeightsStrucure = []
    FlattenedWeights = []
    for i_layer in weights:
        WeightsStrucure.append(i_layer.shape)
        if len(i_layer.shape) == 1 :# 該層權重的shape為一維 e.g. (15,)      
            FlattenedWeights.extend(i_layer)
        else :# 該層權重的shape為二維 e.g. (30, 15)  
            for i_links in i_layer:
                FlattenedWeights.extend(i_links)
    return WeightsStrucure, FlattenedWeights

def self__UnflattenWeights(WeightsStrucure, ModifiedWeights):
    """
    Unflatten(回復成原本的結構) weights  
    
    param WeightsStrucure : 神經網路各層的權重shape包在list中
    param ModifiedWeights : 一維list包含所有meteHeuristic修改過的權重
    return: keras神經網路的權重格式:nparray包在list中
    """
    UnflattenWeights = []
    i_index = 0 
    for i_layer in WeightsStrucure:
        if len(i_layer) == 1 : # 該層權重的shape為一維 e.g. (15,)      
            TempList = ModifiedWeights[i_index:(i_index + i_layer[0])]
            TempList = np.asarray(TempList)
            i_index = i_index + i_layer[0]
        else : # 該層權重的shape為二維 e.g. (30, 15)  
            TempList = ModifiedWeights[i_index:(i_index + (i_layer[0]*i_layer[1]))]
            TempList = np.reshape(TempList, i_layer )
            i_index = i_index + (i_layer[0]*i_layer[1])
        UnflattenWeights.append(TempList)
    return UnflattenWeights   


def self__Recombination(Population_Parents_Weights, Population_Parents_Sigma, rows): #GenerateParents
    """
    Generate New Parents Polulation
    """
    Population_Weights_Recombination = np.zeros(shape = (rows, Population_Parents_Weights.shape[1]))
    Population_Sigma_Recombination = np.zeros(shape = (rows, Population_Parents_Weights.shape[1]))
    for index_row, _ in enumerate( Population_Weights_Recombination ):
        """
        可能可以平行計算
        """
        TwoRowschoiced = np.random.choice(Population_Parents_Weights.shape[0], size=2, replace=False,)
        Parent1Mask = np.random.randint(2, size=Population_Parents_Weights.shape[1])
        Parent2Mask = np.full(shape = Population_Parents_Weights.shape[1], fill_value = 1 )  - Parent1Mask
        
        Population_Weights_Recombination[index_row,:] = (Population_Parents_Weights[TwoRowschoiced] * [Parent1Mask, Parent2Mask]).sum(axis=0)
        Population_Sigma_Recombination[index_row,:] = Population_Parents_Sigma[TwoRowschoiced].mean(axis=0)
    return Population_Weights_Recombination, Population_Sigma_Recombination


def self__score(ModifiedWeights):
    
    """
    Returns objective function value of a state

    :param state: a state
    :return: objective function value of state
    """
    global self_WeightsStrucure, self_KerasModels, self_X_train, self_y_train
    UnflattenedWeights = self__UnflattenWeights(WeightsStrucure = self_WeightsStrucure, ModifiedWeights = ModifiedWeights)
    self_KerasModels.set_weights(UnflattenedWeights)
    test_on_batch = self_KerasModels.test_on_batch(self_X_train, self_y_train) # return ['loss', 'acc']
    #return test_on_batch[0], test_on_batch[1] #0:loss, 1:acc

    return  test_on_batch[1], test_on_batch[0]

def self__best(Population_Child_score):
    """
    Finds the best member of a neighborhood
    :param Population_Child_score: a np array
    :return: the indtex of N best member, N = ParentsSize
    """
    return np.array( Population_Child_score ).argsort()#[::-1]

def self_run(weights, max_steps=5, verbose=10, useOLSReg = False, Population_Parents_Weights=None, Population_Parents_Sigma=None):
    """
    Conducts ES
    :param weights: 
    :param Population_Parents_Weights: 如果想要重複使用Parents Population，則由此傳入Weights
    :param Population_Parents_Sigma: 如果想要重複使用Parents Population，則由此傳入Sigma
    :param max_steps: 
    :param verbose: int which indicates how many iter to show score
    :return: Keras Models, best state and objective function value of best state
    """
    global self_max_steps, self_WeightsStrucure, self_weights, self_best_weight, self_best_Acc, self_best_score, self_ParentsSize, self_InitialSigma, self_cur_steps, self_ChildSize, self_tao
        
    self_max_steps = max_steps
    
    #Step1 initial
    if (Population_Parents_Weights is None) & (Population_Parents_Sigma is None): ##接續上一個phase最好的個體
        
        if isinstance(weights, list)  :          
            self_WeightsStrucure, self_weights = self__FlattenWeights(weights)
            self_best_weight = deepcopy(self_weights)
            self_best_Acc, self_best_score = self__score(self_best_weight)
        else:
            raise TypeError('initial_state must be a list') 
        
        Population_Parents_Weights = np.array([self_weights, self_weights])         
        Population_Parents_Sigma = np.full(shape = (self_ParentsSize, len(self_weights)), fill_value = self_InitialSigma ) 
        Population_Parents_Weights, _ = self__Recombination(Population_Parents_Weights, Population_Parents_Sigma, rows = self_ParentsSize )
        
    
    else: #接續上一個phase的10%最好的個體
        Population_Parents_Weights = Population_Parents_Weights
        Population_Parents_Sigma = Population_Parents_Sigma
        
    self_cur_steps = 1
    while True:   
        #Step2 Child
        ##Discrete Recombination
        Population_Child_Weights, Population_Child_Sigma = self__Recombination(Population_Parents_Weights, Population_Parents_Sigma, rows = self_ChildSize )
        ##mutation1
        RamdonNormalValue = np.random.normal(0, 1, 1)
        RamdonNormalValueDifferent = np.random.normal(0, 1, Population_Child_Sigma.shape)
        Population_Child_Sigma = Population_Child_Sigma * np.exp( (1-self_tao)*RamdonNormalValue + self_tao*RamdonNormalValueDifferent )
        ##mutation2
        Population_Child_Weights = Population_Child_Weights + np.random.normal(0, Population_Child_Sigma, Population_Child_Sigma.shape)
        
        
        # OLS Regression
        if useOLSReg == True:
          for i, i_Child in enumerate(Population_Child_Weights) :
              OLS_Optimized_Weight = self__OLSReg(i_Child)
              #print(OLS_Optimized_Weight,'i:\n', i, Population_Child_Weights[i])
              Population_Child_Weights[i] = OLS_Optimized_Weight
        
        
        #step3 Evaluation
        Population_Child_score = []
        for i_Child in Population_Child_Weights :
            """
            可能可以平行計算
            """
            _, tempScore = self__score(i_Child)
            Population_Child_score.append( tempScore )
        
        #選出最好的ParentsSize個個體做為下一代的親代
        BestIndex = self__best(Population_Child_score)
        BestNIndex = BestIndex[:self_ParentsSize] 
        Population_Parents_Weights = Population_Child_Weights[BestNIndex,:]
        Population_Parents_Sigma = Population_Child_Sigma[BestNIndex,:]
        #選出最好的10%個體做為下一phase的親代
        Best_10Percent_Index = BestIndex[:round(self_ChildSize*0.1)] #Multiphase
        Population_Best_10PercentChild_Weights = Population_Child_Weights[Best_10Percent_Index,:]
        Population_Best_10PercentChild_Sigma = Population_Child_Sigma[Best_10Percent_Index,:] 
        
        #更新best
        best_weight_This_Iter =  Population_Child_Weights[BestNIndex,:][0]
        _, best_score_This_Iter = self__score(best_weight_This_Iter)
        if best_score_This_Iter < self_best_score:
            self_best_weight =  best_weight_This_Iter
            self_best_Acc, self_best_score = self__score(best_weight_This_Iter)
            self_cur_steps = 0#有改善就重新計步
    
        #print process 
        if ((self_cur_steps ) % verbose == 0) and verbose:
            self___str__()#print(self)
            
        self_cur_steps = self_cur_steps + 1
        #step4 check stop criteria
        print(self_cur_steps, max_steps)
        if self_cur_steps > max_steps:
            print( 'Stop: Reach max_steps' )
            break
    return self__UnflattenWeights(WeightsStrucure = self_WeightsStrucure, ModifiedWeights = self_best_weight), self_best_score, Population_Best_10PercentChild_Weights, Population_Best_10PercentChild_Sigma
     

#%% test
KerasModels=model
UseOLSReg=False
InitialSigma = 0.1
ParentsSize = 15
ChildSize = 100
tao = 0.5    

self___init__(model, X_train, y_train, InitialSigma = 0.1, ParentsSize = 15, ChildSize = 100, tao = 0.5)  

useOLSReg =False
max_steps=100
verbose = 1
Population_Parents_Weights=None
Population_Parents_Sigma=None
#%%只用ES run

# Initialize
model = ModelCompile()
Performance(model)
weights = model.get_weights() 
self___init__(model, X_train, y_train, InitialSigma = 0.1, ParentsSize = 15, ChildSize = 100, tao = 0.5)   
weights, ES_Optimized_ObjVal, _, _  = self_run(weights, useOLSReg =False, max_steps=3, verbose = 1)

# Optimize
GlobalBestAccuracy = 10
NoImproveTimes = 0
while True:
  # Gradient-based Optimize
  #model, weights = ModelFit(model=model, weights=weights, epochs=50)

  # ES
  weights, ES_Optimized_ObjVal, _, _ = self_run(weights, max_steps=10, verbose = 1)

  # Stop Criteria  
  if ES_Optimized_ObjVal < GlobalBestAccuracy:
    GlobalBestAccuracy = ES_Optimized_ObjVal
    NoImproveTimes = 0
  else: 
    NoImproveTimes = NoImproveTimes + 1
    if NoImproveTimes == 2:
      break

model.set_weights(weights)  
Performance(model)


#%%

y_score = model.predict(X_test)
y_score

weights




