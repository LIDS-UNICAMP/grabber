import numpy as np
from pyift.livewire import LiveWire
from typing import Optional, Tuple, List
import cv2
from dataclasses import dataclass


@dataclass
class Path:
    coords: Tuple[int, int]
    path: np.ndarray


class Grabber(LiveWire):
    middle: Optional[Path]
    next: Optional[Path]
    previous: Optional[Path]

    def __init__(self, image: np.ndarray, mask: np.ndarray, arc_fun: str = 'exp', epsilon: float = 100,
                 saliency: Optional[np.ndarray] = None, **kwargs):
        """
        TODO
        """
        super().__init__(image, arc_fun, saliency, **kwargs)
        self.epsilon = epsilon
        self.paths = self._process_mask(mask)
        self.middle = None
        self.next = None
        self.previous = None

    def _process_mask(self, mask: np.ndarray) -> List[Path]:
        """
        Parameters
        ----------
        mask: array_like
            Foreground mask.

        Returns
        -------
        List
            List with pairs of anchors coordinates and path sorted by their ordering.
        """
        assert mask.dtype == np.uint8
        mask = np.swapaxes(mask, axis1=0, axis2=1)

        ctr, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        approx = cv2.approxPolyDP(ctr[0], self.epsilon, closed=True)

        ctr = np.squeeze(np.array(ctr[0]), axis=1)
        approx = np.squeeze(np.array(approx), axis=1)

        idx = np.where((ctr == approx[0]).all(axis=1))[0].item()

        ptidx = 0
        splits = [[] for _ in range(len(approx))]
        for _ in range(len(ctr)):
            if ptidx < len(approx) and ctr[idx, 0] == approx[ptidx, 0] and ctr[idx, 1] == approx[ptidx, 1]:
                ptidx += 1
            splits[ptidx - 1].append(ctr[idx, 0] * self.size[1] + ctr[idx, 1])  # swapping to (y, x)
            idx = (idx + 1) % len(ctr)

        paths = []
        for anchor, segment in zip(approx, splits):
            paths.append(Path(tuple(anchor.tolist()),  np.array(segment)))

        self.costs[ctr[:, 0], ctr[:, 1]] = 0
        self.labels[ctr[:, 0], ctr[:, 1]] = True

        return paths

    @staticmethod
    def _assert_position(position: Tuple[int, int]) -> None:
        """
        Asserts position is a tuple of integers.
        """
        if not isinstance(position, tuple):
            raise TypeError('`position` must be a tuple.')
        if not (isinstance(position[0], int) and isinstance(position[1], int)):
            raise TypeError('`position` values must be integers.')

    def _to_index(self, y: int, x: int) -> int:
        """
        Convert coordinates to flattened index.
        """
        return int(self.size[1] * y + x)

    def _find_triplet(self, position: Tuple[int, int]) -> Optional[Tuple[Path, Path, Path, int]]:
        """
        Finds data structure with the same position and its neighbors.
        """
        index = -1
        for i, v in enumerate(self.paths):
            if v.coords[0] == position[0] and v.coords[1] == position[1]:
                index = i
                break

        if index == -1:
            return None

        previous = self.paths[index - 1]
        middle = self.paths[index]
        next = self.paths[(index + 1) % len(self.paths)]
        return previous, middle, next, index

    def select(self, position: Tuple[int, int]) -> bool:
        """
        Selects an anchor point.

        Parameters
        ----------
        position: Tuple[int, int]
            Coordinate (y, x) belonging to an anchor point.
        """
        self._assert_position(position)

        pack = self._find_triplet(position)
        if pack:
            self.previous, self.middle, self.next, _ = pack
            return True
        return False

    def _reset(self, path: np.ndarray) -> None:
        """
        Fills IFT's costs, labels and predecessor map with initilization values.

        Parameters
        ----------
        path: array_like
            Array of coordinates.
        """
        self.costs.flat[path] = np.finfo('d').max
        self.labels.flat[path] = False
        self.preds.flat[path] = -1

    def _draw(self, path: np.ndarray) -> None:
        """
        Fills IFT's costs and labels map with 0 and True, respectively.

        Parameters
        ----------
        path: array_like
            Array of coordinates.
        """
        self.costs.flat[path] = 0
        self.labels.flat[path] = True

    def _opt_path(self, src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[np.ndarray]:
        src = self._to_index(*src)
        dst = self._to_index(*dst)
        self.costs.flat[dst] = np.finfo('d').max
        self.labels.flat[dst] = False
        self.preds.flat[dst] = -1
        return super()._opt_path(src, dst)

    def drag(self, position: Tuple[int, int]) -> None:
        """
        Drag current anchor to new position.

        Parameters
        ----------
        position: Tuple[int, int]
            Anchor's destiny coordinates (y, x).
        """
        if self.middle is None:
            return

        self._assert_position(position)
        if not (0 <= position[0] < self.size[0] and 0 <= position[1] < self.size[1]):
            return

        self._reset(self.previous.path)
        previous_path = self._opt_path(self.previous.coords, position)

        self._reset(self.middle.path)
        middle_path = self._opt_path(position, self.next.coords)

        if middle_path is None or previous_path is None:
            if previous_path is not None:
                self._reset(previous_path)
            if middle_path is not None:
                self._reset(middle_path)
            self._draw(self.previous.path)
            self._draw(self.middle.path)
        else:
            self.middle.coords = position
            self.previous.path = previous_path
            self.middle.path = middle_path

    def confirm(self) -> None:
        """
        Confirms current position.
        """
        self.middle = None
        self.previous = None
        self.next = None

    def add(self, position: Tuple[int, int]) -> int:
        """
        Split contour segment in the desired position, creating an additional anchor.

        Parameters
        ----------
        position: Tuple[int, int]
            Coordinate (y, x) to insert anchor.

        Returns
        -------
        int
            Indexing of added anchor.
        """
        if not isinstance(position, tuple):
            raise TypeError('`position` must be a tuple.')

        new_index = self._to_index(*position)

        for i, path in enumerate(self.paths):
            for j, index in enumerate(path.path):
                if index == new_index:
                    new_path = Path(position, path.path[j:])
                    path.path = path.path[:j]
                    self.paths.insert(i + 1, new_path)
                    return i + 1

        raise ValueError('`position` does not belong to contour.')

    def remove(self, position: Tuple[int, int]) -> None:
        """
        Remove anchor at selected position, computing the optimum-path between its neighbors

        Parameters
        ----------
        position: Tuple[int, int]
            Anchor coordinate (y, x).
        """
        self._assert_position(position)

        pack = self._find_triplet(position)
        if pack is None:
            raise ValueError('`position` not found.')

        previous, current, next, index = pack

        self._reset(previous.path)
        self._reset(current.path)
        self.paths.pop(index)

        previous.path = self._opt_path(previous.coords, next.coords)
