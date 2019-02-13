#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from pathlib import Path


# In[23]:


## This function returns a dictionary of word-embeddings
def get_wordVectors(fpath):
    emb_dict = {}
    numberbatch = open(fpath, 'r').readlines()

    for k, line in enumerate(numberbatch[1:], 1):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype = 'float32')
        emb_dict[word] = vector
        # print (word)
        assert emb_dict[word].shape == (300,)
    
    return emb_dict


# In[24]:


if __name__ == "__main__":
    numberbatch_file = 'numberbatch-en-17.06.txt'
    numberbatch_dir = '../data/'

    fpath = numberbatch_dir + numberbatch_file
    emb_dict = get_wordVectors(fpath)
    embeddings_vectors  = np.stack(list(emb_dict.values()), axis=0)
    print (emb_dict['orange'])


# In[ ]:




