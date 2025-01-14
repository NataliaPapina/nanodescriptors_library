import pandas as pd
from EDA import DataAnalyzer
from nanodesclib.classes import *

df = pd.read_csv('MRI_raw_data.csv')

df = df.dropna(axis='columns', how='all')

df['type'] = df['np_core'].apply(lambda x: assign_class(x)._type)

bar_dict = {'np_shell_1': 'np_shell_1', 'np_shell_2': 'np_shell_2',
            'syn_method': 'syn_method', 'xrd_fracs': 'xrd_fracs', 'emic_exp_type': 'emic_exp_type',
            'emic_shape': 'emic_shape', 'tox_test_type': 'tox_test_type',
            'tox_cell_line': 'tox_cell_line', 'type': 'type_core'}
pie_dict = {'np_shell_3': 'np_shell_3'}
hist_dict = {'np_hydro_size': 'np_hydro_size', 'np_zeta': 'np_zeta',
             'emic_kV': 'emic_kV',  'emic_size': 'emic_size',
             'emic_x_size': 'emic_x_size', 'emic_y_size': 'emic_y_size',
             'emic_z_size': 'emic_z_size', 'squid_temp': 'squid_temp', 'tox_mtt_24': 'tox_mtt_24',
             'squid_h_max': 'squid_h_max', 'squid_sat_mag': 'squid_sat_mag',
             'squid_coerc_f': 'squid_coerc_f', 'squid_rem_mag': 'squid_rem_mag', 'mri_h_val': 'mri_h_val',
             'mri_r1': 'mri_r1', 'mri_r2': 'mri_r2'}

print(DataAnalyzer(df).perform_eda(bar_dict, pie_dict, hist_dict, 'np_core', output_dir='mri_EDA_results'))