# omero-screen-napari

[![License MIT](https://img.shields.io/pypi/l/omero-gallery.svg?color=green)](https://github.com/HocheggerLab/omero-gallery/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/omero-gallery.svg?color=green)](https://pypi.org/project/omero-gallery)
[![Python Version](https://img.shields.io/pypi/pyversions/omero-gallery.svg?color=green)](https://python.org)
[![tests](https://github.com/HocheggerLab/omero-gallery/workflows/tests/badge.svg)](https://github.com/HocheggerLab/omero-gallery/actions)
[![codecov](https://codecov.io/gh/HocheggerLab/omero-gallery/branch/main/graph/badge.svg)](https://codecov.io/gh/HocheggerLab/omero-gallery)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/omero-gallery)](https://napari-hub.org/plugins/omero-gallery)




----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Overview
Plugin to interact with HCS screening data from Omero-Screen data.
Omero-Screen handles metadata, flatfield correction, image segmentation
and cell cycle analysis of EdU labeled IF data.
The resulting data are stored in an Omero database and are required for this plugin to function.

Currently the plugin supports three widgets:
1) A welldata widget to display flatfield corrected well data.
2) Gallery widget to display images from a well in a gallery, choosing segmentation mask, chanels and cell cycle phase
3) A trainingData widget to generate labeled training data for classification.

In development is a widget that initiates training of a CNN for classifiaction and model evaluation.
Unit tests are in development.

## Installation

You can install `omero-gallery` via [pip]:

    pip install omero-gallery



To install latest development version :

    pip install git+https://github.com/HocheggerLab/omero-gallery.git


## Contributing
Robert Zach, Haoran Yue, Alex Herbert and Helfrid Hochegger
@HocheggerLab University of Sussex

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"omero-gallery" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/HocheggerLab/omero-gallery/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
