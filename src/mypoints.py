import numpy as np
from napari.viewer import Viewer
from napari.layers.points import Points
from src.grabber import Grabber


class MyPoints(Points):
    def add(self, coord):
        coord = round(coord[0]), round(coord[1])
        if self.metadata['grabber'].contour[coord]:
            index = self.metadata['grabber'].add(coord)
            self.data = np.insert(self.data, index, np.atleast_2d(coord), axis=0)


def add_my_points(viewer: Viewer, grabber: Grabber, *args, **kwargs) -> MyPoints:
    points = MyPoints(*args, **kwargs)
    points.metadata['grabber'] = grabber
    viewer.add_layer(points)
    return points

