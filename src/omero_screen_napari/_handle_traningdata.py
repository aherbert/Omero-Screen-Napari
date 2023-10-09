from omero_screen_napari.viewer_data_module import viewer_data, cropped_images
from omero_screen_napari.omero_utils import omero_connect
import numpy as np
import tempfile
import omero
from io import BytesIO
import pickle


@omero_connect
def get_saved_data(well_id, classification_name, conn=None):
    well = conn.getObject("Well", well_id)
    for ann in well.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            original_file = ann.getFile()
            file_name = original_file.getName()

            if file_name == f"{classification_name}.pkl":
                print(f"Found file {file_name}. Loading it...")
                data_bytes = read_data(ann)
                data_dict = pickle.loads(data_bytes)
                cropped_images.classifier = data_dict

            elif "npy" in file_name:
                print(f"File {file_name} does not match {classification_name}")
            else:
                print("No previous training data found")


def read_data(ann):
    file_data = ann.getFileInChunks()
    file_bytes = b"".join(file_data)
    print("Previous training data loaded")
    return file_bytes


def store_data(data_array):
    cropped_images.classifier = data_array.tolist()


@omero_connect
def save_trainingdata(well_id, classification_name, conn=None):
    well = conn.getObject("Well", well_id)
    delete_old_data(well, classification_name, conn)
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as temp_file:
            pickle.dump(cropped_images.classifier, temp_file)
            temp_path = temp_file.name
            temp_file.close()

        file_ann = conn.createFileAnnfromLocalFile(
            localPath=temp_path,
            origFilePathAndName=f"{classification_name}.pkl",
            mimetype="application/pkl",
            ns=None,
            desc=None,
        )
        print(f"Uploading new data {file_ann}")
        well.linkAnnotation(file_ann)

    except Exception as e:
        print(f"An error occurred: {e}")



def delete_old_data(well, classification_name, conn):
    for ann in well.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            original_file = ann.getFile()
            file_name = original_file.getName()

            # Check if the file name matches your criteria
            if file_name == f"{classification_name}.pkl":
                print(f"Found file {file_name}. Deleting it...")

                # Delete the FileAnnotation
                conn.deleteObject(ann._obj)
                print(f"Old data {file_name} deleted.")


if __name__ == "__main__":
    sample_data = np.load("../../sample_data/sample_imgs.npy")
    cropped_images.cropped_regions = [
        sample_data[i, :, :, :] for i in range(sample_data.shape[0])
    ]
    cropped_images.cropped_labels = [
        sample_data[i, :, :, :] for i in range(sample_data.shape[0])
    ]
    # cropped_images.classifier = ["unassigned"] * len(
    #     cropped_images.cropped_regions
    # )

    #save_trainingdata(51, 'training_data')

    get_saved_data(51, "training_array")
    print(cropped_images.classifier)
