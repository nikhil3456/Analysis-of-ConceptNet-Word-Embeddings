#!/usr/bin/env python
# coding: utf-8

# In[12]:


import csv


# In[13]:


## returns a dictionary corresponding to the given csv file in path
def get_data(path):
    word_dict = {}
    reader = csv.reader(open(path, "r"), delimiter=",")
    data = list(reader)
    for row in data:
        word_dict[row[0]] = {'pos':row[2]}
    return word_dict


# In[15]:


if __name__ == "__main__":
    file_name="../data/Pictionary Master Target Phrase List - Phrase List.csv"
    print (get_data(file_name))


# In[ ]:




