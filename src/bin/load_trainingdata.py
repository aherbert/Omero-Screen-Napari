#!/usr/bin/env python
from omero_screen_napari.omero_data_singleton import omero_data


def main():
    print("hello world")
    print(omero_data.plate_id)


if __name__ == "__main__":
    main()

