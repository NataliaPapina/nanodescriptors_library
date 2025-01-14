import pandas as pd
import numpy as np
from EDA import DataAnalyzer
from nanodesclib.classes import *
import os

df = pd.read_csv('nanozymes.csv')

bar_list = {'type': 'Type', 'shape': 'Shape', 'ReactionType': 'Reaction Type', 'Subtype': 'Subtype', 'surface': 'surface', 'pol': 'pol',
            'surf': 'surf'}
pie_dict = {'activity': 'Activity', 'Sufrace': 'Surface'}
hist_dict = {'ph': 'pH', 'Syngony': 'Syngony', 'length, nm': 'length', 'width, nm': 'width', 'depth, nm': 'depth',
             'Mw(coat), g/mol': 'Mwcoat', 'Km, mM': 'Km', 'Vmax, mM/s': 'Vmax', 'C min, mM': 'Cmin',
             'C max, mM': 'Cmax', 'C(const), mM': 'Cconst', 'Ccat(mg/mL)': 'Ccat', 'temp, °C': 'temp'}

for i in df.index:
    for col in ['Km, mM', 'Vmax, mM/s', 'C max, mM', 'C(const), mM']:
        try:
            df.loc[i, col] = float((df.loc[i, col]).replace(',', ''))
        except:
            df.loc[i, col] = np.nan

df['Km, mM'] = df['Km, mM'].astype(float)
df['Vmax, mM/s'] = df['Vmax, mM/s'].astype(float)
df['C max, mM'] = df['C max, mM'].astype(float)
df['C(const), mM'] = df['C(const), mM'].astype(float)

replace_dict_formula = {'4N-TiO2': 'N0.17-TiO2',
                        '(Co,Mn)3O4': 'Co3O4-Mn3O4',
                        'BNCuS': 'BN-CuS',
                        'NHMoO3': 'NH-MoO3'}
df['formula'] = df['formula'].replace(replace_dict_formula)

os.makedirs('nanozymes_EDA_results/nanozymes_EDA_results_old_type', exist_ok=True)
DataAnalyzer(df).types_check(df,'formula', output_dir='nanozymes_EDA_results/nanozymes_EDA_results_old_type')

df['type'] = df['formula'].apply(lambda x: assign_class(x)._type)

print(DataAnalyzer(df).perform_eda(bar_list, pie_dict, hist_dict, 'formula', output_dir='nanozymes_EDA_results'))