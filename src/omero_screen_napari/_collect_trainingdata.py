import pickle
import pandas as pd

import omero

from omero_screen_napari.utils import omero_connect
from omero_screen_napari.viewer_data_module import cropped_images

@omero_connect
def get_classification_data(plate_id, classification_name, conn=None):
    plate = conn.getObject("Plate", plate_id)
    classification_data = pd.DataFrame()
    for well in plate.listChildren():
        print(well.getWellPos())
        well_data = load_well_data(well.getId(), classification_name, conn)
        classification_data = pd.concat([classification_data, well_data])
    return classification_data




def load_well_data(well_id, name, conn):
    metadata_dict, label_dict = {}, {}
    well = conn.getObject("Well", well_id)

    for ann in list(well.listAnnotations()):
        if isinstance(ann, omero.gateway.MapAnnotationWrapper):
            metadata_dict = dict(ann.getValue())
        elif isinstance(ann, omero.gateway.FileAnnotationWrapper) and name in ann.getFile().getName():
            data_bytes = read_data(ann)
            label_dict = pickle.loads(data_bytes)
    if label_dict:
        well_df = generate_df(metadata_dict, label_dict)
        well_df['well'] = well.getWellPos()
        print(well_df.head())
        return well_df
    else:
        return pd.DataFrame()



def generate_df(metadata_dict, label_dict):
    df_length = len(label_dict['labels'])
    df = pd.DataFrame([metadata_dict] * df_length)
    df['labels'] = label_dict['labels']
    return df


def read_data(ann):
    file_data = ann.getFileInChunks()
    file_bytes = b"".join(file_data)
    print("Training data loaded")
    return file_bytes




if __name__ == "__main__":
    classification_data = get_classification_data(1430, 'cellcount')
    classification_data.sort_values(by=['well'])
    print(classification_data.head())
    mm231_data = classification_data[classification_data['cell_line'] == 'MM231']
    mm231_data = mm231_data[mm231_data['labels'] != 'unassigned']
    mm231_data.reset_index(drop=True, inplace=True)
    mm231_counts = mm231_data.groupby(['well', 'labels']).count().reset_index()
    mm231_counts.to_csv('~/Desktop/MM231_3d_counts.csv')
    print(mm231_counts)
    HCC1143_data = classification_data[classification_data['cell_line'] == 'HCC1143']
    HCC1143_data = HCC1143_data[HCC1143_data['labels'] != 'unassigned']
    HCC1143_data.reset_index(drop=True, inplace=True)
    HCC1143_counts = HCC1143_data.groupby(['well', 'labels']).count().reset_index()
    HCC1143_counts.to_csv('~/Desktop/HCC1143_3d_counts.csv')
    print(HCC1143_counts)