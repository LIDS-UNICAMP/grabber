from typing import Tuple
import numpy as np
from napari.layers.points import Points


class Anchors(Points):
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
