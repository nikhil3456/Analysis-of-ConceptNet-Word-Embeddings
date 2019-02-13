#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
from word_embedding import get_wordVectors
from get_data import get_data


# In[3]:


def vector_len(v):
    return math.sqrt(sum(x*x for x in v))


# In[4]:


def dot_product(v1, v2):
    assert len(v1) == len(v2)
    return sum(x*y for (x, y) in zip(v1, v2))


# In[5]:


def cosine_similarity(v1, v2):
    """
    Returns the cosine of the angle between the two vectors.
    Results range from -1 (very different) to 1 (very similar).
    """
    return dot_product(v1, v2) / vector_len(v1)*vector_len(v2)


# In[6]:


def sorted_by_similarity(model, words, base_vector):
    """Returns words sorted by cosine distance to a given vector, most similar first"""
    words_with_distance = [(cosine_similarity(base_vector, model[key]), key) for key in words if key in model]
    # We want cosine similarity to be as large as possible (close to 1)
    return sorted(words_with_distance, key=lambda t: t[0], reverse=True)


# In[10]:


def print_related(model, words, text):
    base_word = model[text]
    sorted_words = [
        word for (dist, word) in
            sorted_by_similarity(model, words, base_word)
        ]
    print(', '.join(sorted_words[:7])) #Printing the top 7 neighbours


# In[8]:


file_name="../data/Pictionary Master Target Phrase List - Phrase List.csv"
word_dic = get_data(file_name)

numberbatch_file = 'numberbatch-en-17.06.txt'
numberbatch_dir = '../data/'

fpath = numberbatch_dir + numberbatch_file
emb_dict = get_wordVectors(fpath)


# In[11]:


print_related(emb_dict, word_dic, 'orange')


# In[12]:


print_related(emb_dict, word_dic, 'man')


# In[13]:


print_related(emb_dict, word_dic, 'pepper')


# In[32]:


print_related(emb_dict, word_dic, 'happy')


# In[33]:


print_related(emb_dict, word_dic, 'dance')


# In[34]:


print_related(emb_dict, word_dic, 'trip')


# In[35]:


print_related(emb_dict, word_dic, 'outside')


# In[36]:


print_related(emb_dict, word_dic, 'zero')


# In[38]:


print_related(emb_dict, word_dic, 'number')


# In[ ]:




