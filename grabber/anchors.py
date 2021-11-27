from typing import Tuple
import cv2
import numpy as np
from napari.layers.points import Points
from scipy.spatial import distance

class Anchors(Points):
    def add(self, coord: Tuple[int, int]) -> None:
        coord = round(coord[0]), round(coord[1])
        grabber = self.metadata['grabber']
        
        if self.is_valid(coord):
            coord = self.closest_contour_point(coord)
            index = grabber.add(coord)
            self.data = np.insert(self.data, index, np.atleast_2d(coord), axis=0)

    def is_valid(self, coord: Tuple[int, int]) -> bool:
        grabber = self.metadata['grabber']
        if grabber is None:
            return False
        y, x = grabber.costs.shape
        return 0 <= coord[0] < y and 0 <= coord[1] < x
    
    def is_on_contour(self, coord: Tuple[int, int]) -> bool:
        grabber = self.metadata['grabber']
        if grabber is None:
            return False
        return grabber.contour[coord] == True

    def closest_contour_point(self, coord) -> Tuple[int, int]:
        grabber = self.metadata['grabber']
        contour = np.array(np.nonzero(grabber.contour)).T

        closest_index = distance.cdist(contour, [coord]).argmin()
        coord = contour[closest_index]
        coord = int(coord[0]), int(coord[1])

        return coord