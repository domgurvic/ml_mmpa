
import os
import sys
import pandas as pd
import re
from tqdm import tqdm



def split_transition(df, col):
    df['LHS'] = [re.split('>>',df[col].values[i])[0] for i in range(len(df)) ]
    df['RHS'] = [re.split('>>',df[col].values[i])[1] for i in range(len(df)) ]

    return df

def add_column(nom,col, df, property):

    prop=[]

    for no in tqdm(range(len(df))):
        val_a = property[property.columns[1]].iloc[df[nom].iloc[no]-1]
        prop.append(val_a)

    df[col] = prop

    return df

def add_property(index_file, property_file): 

    print('Adding property back into index') 
    
    property=pd.read_csv(property_file, sep=' ')
    
    index=pd.read_csv(index_file,  header = None, sep='\t')
    
    print('Found pairs: ', len(index))

    index_temp = add_column(2, 'measurement_A', index, property)
    index_fin = add_column(3, 'measurement_B', index_temp, property)
    index_fin['measurement_delta'] = index_fin['measurement_B'] - index_fin['measurement_A']
    
    return index


if __name__ == '__main__':
    file = add_property(sys.argv[1],sys.argv[2])
    df = split_transition(file, 4)
    df.rename(columns = {0:'compound_structure_A', 1:'compound_structure_B', 2:'idsmiles_A', 3:'idsmiles_B', 4:'smirks',  5:'common_core'}, inplace = True)
    print('saving final file: ', str(os.path.splitext(sys.argv[1])[0]+'_final.csv'))
    df.to_csv(str(os.path.splitext(sys.argv[1])[0]+'_final.csv'), index=False)
