#Example Scikit-Learn Custom Transformer 

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import copy
import numpy as np 
import pandas as pd
from random import shuffle

#This is use by Function Transformer
def round_time(x):
    #import pdb
    #pdb.set_trace()
    for c in x.columns.values:
        x[c] = pd.to_datetime(x[c])
        x[c] = x[c].dt.round('H').dt.hour
    return x.values 

#Custom transformer implementation
class Cat2VecTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, windows, min_count, cat_feature_size):
        self.windows = windows
        self.min_count = min_count
        self.cat_feature_size = cat_feature_size
        self.model = None
        self.vocab = None
        #self.categories = categories
  
  def __convert_to_cat(self, df):    
    data = copy.deepcopy(df)
    data.reset_index(drop=True, inplace=True)
    for c in list(data.columns.values):
        data[c] = data[c].astype('category')
        data[c].cat.categories = ["%s %s" % (c,g) for g in data[c].cat.categories]        
    return data
  
  def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        #import pdb
        #pdb.set_trace()
        return self.partial_fit(X, y)
    
  def partial_fit(self, X, y=None):
    def shuffle_list_item(alist):
        shuffle(alist)
        return alist
    
    cat_list = self.__convert_to_cat(X).values.tolist()
    cat_list = list(map(shuffle_list_item, cat_list))
            
    if self.model is None:
      self.model = Word2Vec(cat_list, size=self.cat_feature_size, window=self.windows, min_count=self.min_count, workers=1)      
    else:      
      self.model.train(cat_list, total_examples=len(X))
    
    self.vocab = set(self.model.wv.index2word)
    return self
  
  def transform(self, X):
    #vocab = set(self.model.wv.index2word)
    all_cat_vec = np.zeros((X.shape[0], X.shape[1] * self.cat_feature_size), dtype="float64")
    
    #cats_df.reset_index(drop=True, inplace=True)
    X_cat = self.__convert_to_cat(X)
    for index, row in X_cat.iterrows():        
      for catIndex, catItem in enumerate(row):
        if catItem in self.vocab:
          startIndex = catIndex * self.cat_feature_size
          endIndex = (catIndex + 1) * self.cat_feature_size
          all_cat_vec[index, startIndex : endIndex] = self.model[catItem]
    
    return all_cat_vec
