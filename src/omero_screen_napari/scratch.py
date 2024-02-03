import pandas as pd
import numpy as np

df = pd.read_csv('~/Desktop/test_plate03_final_data_cc.csv', index_col=0)
channels = {
    'channel_0': 'DAPI',
    'channel_1': 'Tub',
    'channel_2': 'p21',
    'channel_3': 'EdU',
}
intensity_dict = {}
for key, value in channels.items():
    
    max_value = df[f"intensity_max_{value}_nucleus"].median()
    min_value = df[f"intensity_min_{value}_nucleus"].median()
    intensity_dict[int(key.split('_')[1])] = (max_value, min_value)
print(intensity_dict)
