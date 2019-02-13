#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import os
import math
from word_embedding import get_wordVectors
from get_data import get_data
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


# In[29]:


file_name="../data/Pictionary Master Target Phrase List - Phrase List.csv"
word_dic = get_data(file_name)

numberbatch_file = 'numberbatch-en-17.06.txt'
numberbatch_dir = '../data/'

fpath = numberbatch_dir + numberbatch_file
emb_dict = get_wordVectors(fpath)

FOLDER_PATH = "/home/nikhil/conceptNet/code"

VOCAB_SIZE = len(word_dic)
EMBEDDING_DIM = emb_dict['orange'].shape[0]


# In[30]:


word_vectors = []
for key in word_dic:
    if key in emb_dict: word_vectors.append(emb_dict[key])
    else: word_vectors.append(emb_dict['##'])

word_vectors = np.stack(word_vectors, axis=0)
# print (word_vectors[0])
# print (word_vectors.shape)


# In[31]:


embeddings = emb_dict
embeddings_vectors  = np.stack(list(embeddings.values()), axis=0)

# print (embeddings_vectors[0])
# print (embeddings_vectors.shape)


# In[32]:


# For visualizing words in pictionary master List, uncomment below line
vectors = word_vectors
words = word_dic

print (vectors.shape)
emb = tf.Variable(vectors, name='word_embedding')
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, "model_dir/model.ckpt")
    print("model saved in path: %s" % save_path)


# In[33]:


words = '\n'.join(list(words.keys()))

with open(os.path.join('model_dir', 'metadata.tsv'), 'w') as f:
    f.write(words)


# In[ ]:




