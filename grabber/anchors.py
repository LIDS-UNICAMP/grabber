from typing import Tuple
import numpy as np
from napari.layers.points import Points
from napari.utils.events import Event
from scipy.spatial import distance
import warnings


class Anchors(Points):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.add(
            contour=Event,
        )

    def add(self, coord: Tuple[int, int]) -> None:
        coord = round(coord[0]), round(coord[1])
        grabber = self.metadata['grabber']
        
        if self.is_valid(coord):
            coord = self.closest_contour_point(coord)
            index = grabber.add(coord)
            self.data = np.insert(self.data, index, np.atleast_2d(coord), axis=0)

    def remove_selected(self) -> None:
        if 'grabber' in self.metadata:
            grabber = self.metadata['grabber']
            for i in sorted(list(self.selected_data), reverse=True):
                if len(grabber.paths) <= 2:
                    warnings.warn('Number of anchors cannot be less than 2.')
                    break

                pt = grabber.paths[i].coords
                grabber.remove(pt)

            self.events.contour()

        super().remove_selected()

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