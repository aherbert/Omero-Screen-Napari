"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING, Annotated
from enum import Enum
from magicgui import magic_factory
from magicgui.widgets import FloatSlider
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


class Options(Enum):
    Option1 = "Option 1"
    Option2 = "Option 2"
    Option3 = "Option 3"


@magic_factory(call_button="Enter")
def example_magic_widget(
    dropdown: Options = Options.Option1,
):
    print(f"Dropdown input received: {dropdown.value}")
    option_value = dropdown.value
    perform_action_widget = perform_action(selected_option=option_value)
    perform_action_widget.show()


@magic_factory(call_button="Perform Action")
def perform_action(selected_option: str = "Default Option"):
    # Your code here to perform the action based on the selected option
    print(f"the 2nd widget received: {selected_option}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
