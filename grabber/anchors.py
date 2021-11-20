from typing import Tuple
import cv2
import numpy as np
from napari.layers.points import Points
from scipy.spatial import distance

class Anchors(Points):
    def add(self, coord: Tuple[int, int]) -> None:
        coord = round(coord[0]), round(coord[1])
        grabber = self.metadata['grabber']
        
        if self.is_valid(coord, on_contour=False):
            coord = self.closest_contour_point(coord)
            index = grabber.add(coord)
            self.data = np.insert(self.data, index, np.atleast_2d(coord), axis=0)

    def is_valid(self, coord: Tuple[int, int], on_contour: bool) -> bool:
        grabber = self.metadata['grabber']
        if grabber is None:
            return False
        y, x = grabber.costs.shape
        return 0 <= coord[0] < y and 0 <= coord[1] < x and grabber.contour[coord] == on_contour

    def closest_contour_point(self, coord) -> Tuple[int, int]:
        grabber = self.metadata['grabber']
        contours, _ = cv2.findContours(grabber.contour.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = sorted(contours, key=cv2.contourArea)[-1]
        contour = contour.reshape((contour.shape[0], contour.shape[2]))
        contour[:,[0, 1]] = contour[:,[1, 0]]

        closest_index = distance.cdist(contour, [coord]).argmin()
        coord = contour[closest_index]
        coord = int(coord[0]), int(coord[1])

        return coord
