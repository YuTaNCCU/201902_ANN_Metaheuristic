#%% 創建Class TabuSearch
########################################################################
#參考來源： https://github.com/100/Solid/edit/master/Solid/TabuSearch.py#
########################################################################
import random
from string import ascii_lowercase
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax
from keras import backend as K
from keras.models import Sequential 
import numpy as np


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
    
    def __init__(self, KerasModels, InitialSigma = 0.1, ParentsSize = 15, ChildSize = 100, tao = 0.5):
        """
        :param KerasModels: a Keras model, like keras.engine.sequential.Sequential
        :param weights: initial weights, should be a Keras model weight
        :param max_steps: maximum number of steps to run algorithm for
        """
        self.KerasModels = KerasModels
 
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
        return ('ES: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST MEMBER: \n %s ......\n' +
                'BEST SCORE%f \n') % \
               (self.cur_steps, str(self.best_weight)[:100], self.best_score)

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
        return np.array( Population_Child_score ).argsort()[::-1][:self.ParentsSize]
    
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
        test_on_batch = self.KerasModels.test_on_batch(X_train, y_train, sample_weight=None) # return ['loss', 'acc']
        return test_on_batch[1]

    def run(self, weights, max_steps=5, verbose=10):
        """
        Conducts ES
        :param weights: 
        :param max_steps: 
        :param verbose: int which indicates how many iter to show score
        :return: Keras Models, best state and objective function value of best state
        """
        
        if isinstance(weights, list)  :
            self.WeightsStrucure, self.weights = self._FlattenWeights(weights)
        else:
            raise TypeError('initial_state must be a list') 
        
        self.max_steps = max_steps
        
        #Step1 initial             
        Population_Parents_Weights = np.array([self.weights, self.weights])         
        Population_Parents_Sigma = np.full(shape = (self.ParentsSize, len(self.weights)), fill_value = self.InitialSigma ) 
        Population_Parents_Weights, _ = self._Recombination(Population_Parents_Weights, Population_Parents_Sigma, rows = self.ParentsSize )
           
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
            
            #step3 Evaluation
            Population_Child_score = []
            for i_Child in Population_Child_Weights :
                """
                可能可以平行計算
                """
                Population_Child_score.append( self._score(i_Child) )
            BestNIndex = self._best(Population_Child_score)
            Population_Parents_Weights = Population_Child_Weights[BestNIndex,:]
            Population_Parents_Sigma = Population_Child_Sigma[BestNIndex,:]
            
            #step4 check stop criteria
            if self.cur_steps > max_steps:
                print( 'Stop: Reach max_steps' )
                break
            
            self.best_weight =  Population_Child_Weights[BestNIndex,:][0]
            self.best_score = self._score(Population_Child_Weights[BestNIndex,:][0])
           
            #print process 
            if ((self.cur_steps ) % verbose == 0) and verbose:
               print(self)
               
            self.cur_steps = self.cur_steps + 1
        return Population_Child_Weights[BestNIndex,:][0], self._score(Population_Child_Weights[BestNIndex,:][0])

#%% Dataset split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

#資料集是以dictionary的形式存在
cancer = load_breast_cancer()
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

X = df_feat.iloc[:, ].values
y = cancer['target']

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

#X_train.shape,X_test.shape,y_train.shape,y_test.shape

#% 創建神經網路
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential, Model as keras_models_Model

model = Sequential()
model.add(Dense(1, activation='relu', input_shape=(30,)))
#model.add(Dense(5, activation='relu'))
#model.add(Dense(2, activation='relu', name = 'IntermediateLayer'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#%% run
        
# initialize
weights = model.get_weights() 
MyES = ES(model, InitialSigma = 0.1, ParentsSize = 15, ChildSize = 100, tao = 0.5)   
ES_Optimized_Weights, ES_Optimized_ObjVal  = MyES.run(weights, max_steps=5, verbose = 1)
#%% run2
# gradient-based optimize
model.fit(X_train, y_train, epochs=5, batch_size=32)
weights = model.get_weights() 

# ES
ES_Optimized_Weights, ES_Optimized_ObjVal  = MyES.run(weights, max_steps=10, verbose = 1)

#%% obtain the output of an intermediate layer
#https://keras.io/getting-started/faq/?fbclid=IwAR3Zv35V-vmEy85anudOrlxCExXYwyG6cRL1UR0AaLPU6sZEoBjsbX-8LXQ#how-can-i-obtain-the-output-of-an-intermediate-layer

layer_name = 'IntermediateLayer'
intermediate_layer_model = keras_models_Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data_x)

"""
How can I obtain the output of an intermediate layer?
One simple way is to create a new Model that will output the layers that you are interested in:

from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
Alternatively, you can build a Keras function that will return the output of a certain layer given a certain input, for example:

from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
Similarly, you could build a Theano and TensorFlow function directly.

Note that if your model has a different behavior in training and testing phase (e.g. if it uses Dropout, BatchNormalization, etc.), you will need to pass the learning phase flag to your function:

get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]

"""
#%%
#%%
#%%
#%%
#%%待續
向量式運算找鄰近解
Diverse: (meta + 最後一層OLS)執行幾次後 > Intense: 換成SGD
找一個新的演算法

#%% Path Relinking
#https://github.com/priitj/grils-t/blob/master/relink.py

#grasp
#http://www2.imm.dtu.dk/courses/02719/grasp/grasp.pdf  簡報介紹

#https://grasp.readthedocs.io/en/latest/index.html 套件

#https://www.google.com/search?q=grasp+metaheuristic&rlz=1C1GCEU_zh-twTW836TW836&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiglIa3xq7hAhUY5bwKHSQoCqQQ_AUIDigB&biw=1280&bih=631
#%% flatten weights & Unflatten(回復成原本的結構) weights
def _FlattenWeights(weights):
    """
    flatten weights
    
    param weights: keras神經網路的權重格式:nparray包在list中
    return WeightsStrucure : 神經網路各層的權重shape包在list中，unflatten時會用到
    return FlattenedWeights : 一維list包含所有的權重
    """
    WeightsStrucure = []
    FlattenedWeights = []
    for i_layer in weights:
        print(i_layer.shape)
        WeightsStrucure.append(i_layer.shape)
        if len(i_layer.shape) == 1 :# 該層權重的shape為一維 e.g. (15,)      
            FlattenedWeights.extend(i_layer)
        else :# 該層權重的shape為二維 e.g. (30, 15)  
            for i_links in i_layer:
                FlattenedWeights.extend(i_links)
    return WeightsStrucure, FlattenedWeights

def _UnflattenWeights(WeightsStrucure, ModifiedWeights):
    """
    Unflatten(回復成原本的結構) weights  
    
    param WeightsStrucure : 神經網路各層的權重shape包在list中
    param ModifiedWeights : 一維list包含所有meteHeuristic修改過的權重
    return: keras神經網路的權重格式:nparray包在list中
    """
    import numpy as np
    UnflattenWeights = []
    i_index = 0
    for i_layer in WeightsStrucure:
        print(i_layer)
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

WeightsStrucure, FlattenedWeights = FlattenWeights(weights)
###
#MetaHeuristic return optimized weights
##
ModifiedWeights= FlattenedWeights        #ModifiedWeights為meteHeuristic修改過的權重
UnflattenWeights = _UnflattenWeights(WeightsStrucure, ModifiedWeights)