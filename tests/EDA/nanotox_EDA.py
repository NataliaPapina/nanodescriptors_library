import pandas as pd
from EDA import DataAnalyzer
from nanodesclib.assign_class import *

df = pd.read_csv('nanotox.csv')

df['type'] = df['material'].apply(lambda x: assign_class(x)._type)

bar_list = {'type': 'Type', 'shape': 'Shape', 'coat/functional group': 'coat functional group', 'synthesismethod': 'synthesismethod',
            'surface charge': 'surface charge', 'cell source': 'cell source', 'cell tissue': 'cell tissue',
            'cell morphology': 'cell morphology', 'cell age': 'cell age',
            'test': 'test', 'test indicator': 'test indicator', 'Cell type': 'Cell type'}
pie_dict = {'human/animal': 'human_animal'}
hist_dict = {'size in medium (nm)': 'size in medium', 'zeta in medium (mV)': 'zeta in medium',
             'no of cells (cells/well)': 'no of cells',  'time (hr)': 'time hr',
             'concentration (ug/ml)': 'concentration', 'viability (%)': 'viability',
             'core size (nm)': 'core size', 'surface area': 'surface area',
             'Hydrodynamic diameter (nm)': 'Hydrodynamic diameter', 'Zeta potential (mV)': 'Zeta potential',
             'Molecular weight (g/mol)': 'Molecular weight'}

print(DataAnalyzer(df).perform_eda(bar_list, pie_dict, hist_dict, 'material', output_dir='nanotox_EDA_results'))