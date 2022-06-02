#!/usr/bin/env python
# coding: utf-8

# In[2]:


# perform mmpa on file: csv ['smiles', 'property']


print('Importing libraries')

import os
import sys
sys.path.append("/homes/dgurvic/software/miniconda3/envs/jupt_test/lib/python3.7/site-packages/")
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

def mmpa(smiles_file_location, name):
    
    print('initializing mmpa on:', name )


    name_csv = name + '.csv'
    name_smi = name + '.smi'
    name_frag = name + '.fragments'
    
    file = pd.read_csv(smiles_file_location)
    
    file["ID"] = file.index + 1

    print('first columns should be smiles: ', file.columns[0])

    first_file = file[[file.columns[0], 'ID']].to_csv('out/' + name_smi,  sep = ' ', index=False, header=False)
    
    print('Second column should be property: ', file.columns[1])


    second_file = file[['ID', file.columns[1]]]
    
    print('saving property file: ','out/' +  name + '_property' + '.csv')


    second_file.to_csv('out/' + name + '_property' + '.csv',  sep = ' ', index = False)
   
#    os.system('echo here  ${name_frag}')

    #print(name_frag)

#    os.system(' python mmpdb/mmpdb fragment $name_smi -o $name_frag')
    
#    os.system(' python mmpdb/mmpdb index $name_frag -o $name_csv')
    
    return


if __name__ == '__main__':
    mmpa(sys.argv[1],sys.argv[2] )



# In[ ]:




