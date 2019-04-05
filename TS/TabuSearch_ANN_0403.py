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


class TabuSearch:
    """
    Conducts tabu search
    """
    __metaclass__ = ABCMeta

    cur_steps = None

    tabu_size = None
    tabu_list = None

    initial_state = None
    current = None
    best = None

    max_steps = None
    max_score = None
    
    KerasModels = None
    WeightsStrucure = None
    
    def FlattenWeights(self, weights):
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

    def UnflattenWeights(self, WeightsStrucure, ModifiedWeights):
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
    
    def __init__(self, KerasModels, initial_state, tabu_size, max_steps, max_score=None):
        """
        :param KerasModels: a Keras model, like keras.engine.sequential.Sequential
        :param initial_state: initial state, should implement __eq__ or __cmp__
        :param tabu_size: number of states to keep in tabu list
        :param max_steps: maximum number of steps to run algorithm for
        :param max_score: score to stop algorithm once reached
        """
        self.KerasModels = KerasModels
        
        if isinstance(initial_state, list):
            WeightsStrucure, FlattenedWeights = self.FlattenWeights(initial_state)
            self.WeightsStrucure = WeightsStrucure
            self.initial_state = FlattenedWeights
            self.best = FlattenedWeights
        else:
            raise TypeError('initial_state must be a list')

        if isinstance(tabu_size, int) and tabu_size > 0:
            self.tabu_size = tabu_size
        else:
            raise TypeError('Tabu size must be a positive integer')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise TypeError('Maximum steps must be a positive integer')

        if max_score is not None:
            if isinstance(max_score, (int, float)):
                self.max_score = float(max_score)
            else:
                raise TypeError('Maximum score must be a numeric type')
        

    def __str__(self):
        return ('TABU SEARCH: \n' +
                'CURRENT STEPS: %d \n' +
                'BEST SCORE: %f \n' +
                'BEST MEMBER: \n %s ......\n\n') % \
               (self.cur_steps, self._score(self.best), str([round(i,5) for i in self.best])[:100])

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm

        :return: None
        """
        self.cur_steps = 0
        
        #deque(maxlen=N) 创建了一个固定长度的队列，当有新的记录加入而队列已满时会自动移动除最老的那条记录
        self.tabu_list = deque(maxlen=self.tabu_size) 
        self.current = self.initial_state
        self.best = self.initial_state

    @abstractmethod
    def _score(self, state):
        """
        Returns objective function value of a state

        :param state: a state
        :return: objective function value of state
        """
        pass

    @abstractmethod
    def _neighborhood(self):
        """
        Returns list of all members of neighborhood of 
        state, given self.current

        :return: list of members of neighborhood
        """
        pass

    def _best(self, neighborhood):
        """
        Finds the best member of a neighborhood

        :param neighborhood: a neighborhood
        :return: best member of neighborhood
        """
        return neighborhood[argmax([self._score(x) for x in neighborhood])]

    def run(self, verbose=True):
        """
        Conducts tabu search

        :param verbose: indicates whether or not to print progress regularly
        :return: Keras Models, best state and objective function value of best state
        """
        self._clear()
        for i in range(self.max_steps):
            self.cur_steps += 1

            if ((i + 1) % 1 == 0) and verbose:
                print(self)

            neighborhood = self._neighborhood()
            neighborhood_best = self._best(neighborhood)

            while True:
                #neighborhood中是否全部在tabu list中
                if all([self._IfInTabuList(OneNeighbor) for OneNeighbor in neighborhood]): #所有元素是否都为TRUE，如果是返回True，否则返回False。
                    print("TERMINATING - NO SUITABLE NEIGHBORS")
                    return self.KerasModels,  self.best, self._score(self.best)
                
                #neighborhood_best是否在tabu list中
                if self._IfInTabuList(neighborhood_best): 
                    #使用 Aspiration 
                    if self._score(neighborhood_best) > self._score(self.best):
                        self.tabu_list.append(neighborhood_best)
                        self.best = deepcopy(neighborhood_best)
                        break
                    #換一個neighbor作為neighborhood_best
                    else:
                        neighborhood.remove(neighborhood_best)
                        neighborhood_best = self._best(neighborhood)
                #從current走到neighborhood_best
                else:
                    self.tabu_list.append(neighborhood_best)
                    self.current = neighborhood_best
                    #更新self.best
                    if self._score(self.current) > self._score(self.best):
                        self.best = deepcopy(self.current)
                    break

            if self.max_score is not None and self._score(self.best) > self.max_score:
                print("TERMINATING - REACHED MAXIMUM SCORE")
                return self.best, self._score(self.best)
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.KerasModels, self.best, self._score(self.best)

#%% 繼承class TabuSearch與客製化函數:_neighborhood、_scoreFlattenWeights、FlattenWeights、UnflattenWeights
class TabuSearchCustomized(TabuSearch):
    """
    Tries to get  
    """
    def _score(self, ModifiedWeights):
        """
        obj. fun.
        """ 
        UnflattenedWeights = self.UnflattenWeights(WeightsStrucure = self.WeightsStrucure, ModifiedWeights = ModifiedWeights)
        self.KerasModels.set_weights(UnflattenedWeights)
        test_on_batch = self.KerasModels.test_on_batch(data_x, data_y, sample_weight=None) # return ['loss', 'acc']
        return test_on_batch[1]
    
    def _neighborhood(self):
        """
        return a list of neighbors (with length of _NumOfNeighbor) from current state
        """
        ##################
        _NumOfNeighbor=50
        ##################
        
        neighborhood = []        
        for _ in range( _NumOfNeighbor ): #鄰居數
            OneNeighbor = []
            #找鄰居的方法:
            for i in self.current:  
                OneNeighbor.append( i * random.uniform(-1, 1 ) )
            neighborhood.append(OneNeighbor) #feasible的鄰居放入備選list（neighborhood）
            # neighborhood.append( [i * random.uniform(-1, 1 )  for i in self.current]]) 
        return neighborhood

    
    def _IfInTabuList( self, OneNeighbor ):
        """
        determine whether a OneNeighbor is in tabulist
        :param OneNeighbor: a list of weights
        :return: True if in the tabu list, False if not in the tabu list, 
        """
        ####################################
        Threohold__IfInTabuList_MAPE = 0.5
        ####################################
        for i_tabu_list in self.tabu_list:
            
            y_true = np.asarray(i_tabu_list)
            y_pred = np.asarray(OneNeighbor)
 
            MAPE = np.mean(np.abs((y_true - y_pred)/(y_true+0.0000001)))
            if MAPE <=  Threohold__IfInTabuList_MAPE :#MAPE
                print('TABU!!!!!!!!!!!', MAPE)
                return True #在tabu list中
        return False #未在tabu list中
#%% Loading Data
from sklearn.datasets import load_breast_cancer
return_X_y = load_breast_cancer()           
data_x = return_X_y.data #569, 30
data_y = return_X_y.target #569, 1 #0惡性 1良性

#%% 創建神經網路
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential, Model as keras_models_Model

model = Sequential()
model.add(Dense(15, activation='relu', input_shape=(30,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu', name = 'IntermediateLayer'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(data_x, data_y, epochs=5, batch_size=1024)

weights = model.get_weights()
###
#MetaHeuristic return optimized weights
##
model.set_weights(weights)
test_on_batch = model.test_on_batch(data_x, data_y, sample_weight=None) # return ['loss', 'acc']
test_on_batch[1]
#%% 創建Tabu Search物件(Object)
#print('Initial Obj. Val.: '  )
#print('Initial Solution: \n' , '\n\n')

TSRun = TabuSearchCustomized(model, weights, tabu_size=10 , max_steps=10, max_score=None)         

TS_Optimized_Model, TS_Optimized_Weights, TS_Optimized_ObjVal  = TSRun.run()

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
Diverse: meta執行幾次後>Intense: 換成SGD
meta結合最後一層LS
找一個新的演算法

#%% Path Relinking
#https://github.com/priitj/grils-t/blob/master/relink.py

#grasp
#http://www2.imm.dtu.dk/courses/02719/grasp/grasp.pdf  簡報介紹

#https://grasp.readthedocs.io/en/latest/index.html 套件

#https://www.google.com/search?q=grasp+metaheuristic&rlz=1C1GCEU_zh-twTW836TW836&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiglIa3xq7hAhUY5bwKHSQoCqQQ_AUIDigB&biw=1280&bih=631
#%% flatten weights & Unflatten(回復成原本的結構) weights
def FlattenWeights(weights):
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

def UnflattenWeights(WeightsStrucure, ModifiedWeights):
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
UnflattenWeights = UnflattenWeights(WeightsStrucure, ModifiedWeights)
#%%

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%