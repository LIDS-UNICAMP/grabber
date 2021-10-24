from typing import Tuple

import numpy as np
from napari.viewer import Viewer
from napari.layers.points import Points
from src.grabber import Grabber


class MyPoints(Points):
    def add(self, coord: Tuple[int, int]) -> None:
        coord = round(coord[0]), round(coord[1])
        grabber = self.metadata['grabber']
        y, x = grabber.costs.shape
        if self.is_valid(coord, on_contour=True):
            index = grabber.add(coord)
            self.data = np.insert(self.data, index, np.atleast_2d(coord), axis=0)

    def is_valid(self, coord: Tuple[int, int], on_contour: bool) -> bool:
        grabber = self.metadata['grabber']
        y, x = grabber.costs.shape
        return 0 <= coord[0] < y and 0 <= coord[1] < x and grabber.contour[coord] == on_contour


def add_my_points(viewer: Viewer, grabber: Grabber, *args, **kwargs) -> MyPoints:
    points = MyPoints(*args, **kwargs)
    points.metadata['grabber'] = grabber
    viewer.add_layer(points)
    return points


@MyPoints.bind_key('Space')
def hold_to_pan_zoom(layer):
    """Hold to pan and zoom in the viewer."""
    if layer.mode != "pan_zoom":
        # on key press
        prev_mode = layer.mode
        prev_selected = layer.selected_data.copy()
        layer.mode = "pan_zoom"

        yield

        # on key release
        layer.mode = prev_mode
        layer.selected_data = prev_selected
        layer._set_highlight()

