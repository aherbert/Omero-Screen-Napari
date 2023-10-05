from omero_screen_napari.viewer_data_module import viewer_data, cropped_images
from omero_screen_napari.omero_utils import omero_connect
import numpy as np
import tempfile
import omero
from io import BytesIO


@omero_connect
def get_saved_data(well_id, classification_name, conn=None):
    well = conn.getObject("Well", well_id)
    for ann in well.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            original_file = ann.getFile()
            file_name = original_file.getName()

            if file_name == f"{classification_name}.npy":
                print(f"Found file {file_name}. Loading it...")
                data_array = read_data(ann)
                store_data(data_array)
            elif "npy" in file_name:
                print(f"File {file_name} does not match {classification_name}")
            else:
                print("No previous training data found")


def read_data(ann):
    file_data = ann.getFileInChunks()
    file_bytes = b"".join(file_data)
    array_data = np.load(BytesIO(file_bytes), allow_pickle=True)
    print("Previous training data loaded")
    return array_data


def store_data(data_array):
    cropped_images.classifier = data_array.tolist()


@omero_connect
def save_trainingdata(well_id, classification_name, conn=None):
    well = conn.getObject("Well", well_id)
    delete_old_data(well, classification_name, conn)
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            np.save(temp_path, cropped_images.classifier)
            temp_file.close()

        file_ann = conn.createFileAnnfromLocalFile(
            localPath=temp_path,
            origFilePathAndName=f"{classification_name}.npy",
            mimetype="application/npy",
            ns=None,
            desc=None,
        )
        print(f"Uploading new data {file_ann}")
        well.linkAnnotation(file_ann)

    except Exception as e:
        print(f"An error occurred: {e}")


# def transform_data():
#     # Create an empty object array to hold all the data
#     combined_array = np.empty(
#         (3, len(cropped_images.cropped_regions)), dtype=object
#     )
#
#     combined_array[0, :] = cropped_images.cropped_regions
#     combined_array[1, :] = cropped_images.cropped_labels
#     combined_array[2, :] = cropped_images.classifier
#     return combined_array


def delete_old_data(well, classification_name, conn):
    for ann in well.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            original_file = ann.getFile()
            file_name = original_file.getName()

            # Check if the file name matches your criteria
            if file_name == f"{classification_name}.npy":
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
