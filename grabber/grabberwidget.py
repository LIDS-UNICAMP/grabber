
from typing import Optional, Tuple
from numpy.typing import ArrayLike

import napari
from napari.layers import Image, Labels
from magicgui.widgets import Container, create_widget, PushButton

import warnings
import numpy as np
import cv2
from .grabber import Grabber
from .anchors import Anchors


class GrabberWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._grabber: Optional[Grabber] = None

        self._contour_layer = self._viewer.add_labels(
            np.zeros((1,1), dtype=int), color={1: 'cyan'}, name='Contour', opacity=1.0
        )
        self._anchors_layer = self._create_anchors()

        self._image_layer = create_widget(annotation=Image, label='Image')
        self.append(self._image_layer)
        
        self._labels_layer = create_widget(annotation=Labels, label='Labels')
        self.append(self._labels_layer)

        self._label_selection = create_widget(1, label='Selected Label')
        self.append(self._label_selection)
        
        self._preprocess = create_widget(True, label='Preprocess', options=dict(
            tooltip='Convert RGB images to LAB (if possible) and apply bilateral filtering.'
        ))
        self.append(self._preprocess)

        self._load_button = PushButton(text='Load')
        self._load_button.changed.connect(self._on_load)
        self.append(self._load_button)

        set_button_status = lambda _: setattr(self._load_button, 'enabled', self._validate_load())
        self._image_layer.changed.connect(set_button_status)
        self._labels_layer.changed.connect(set_button_status)

        self._sigma = create_widget(
            30.0, widget_type='FloatSlider', label='Sigma',
            options=dict(min=0.01, max=255.0,
                tooltip='Lower values makes contour adheres more to the image (and noise) boundaries.'
            )
        )
        self._sigma.changed.connect(self._update_sigma)
        self.append(self._sigma)

        self._epsilon = create_widget(
            25.0, widget_type='FloatSlider', label='Epsilon',
            options=dict(min=1.0, max=500.0,
                         tooltip='Smaller values generates more anchor points.'
            )
        )
        self._epsilon.changed.connect(self._update_epsilon)
        self.append(self._epsilon)

        self._confirm_button = PushButton(text='Confirm', enabled=False)
        self._confirm_button.changed.connect(self._on_confirm)
        self.append(self._confirm_button)

        disable_confirm = lambda _: setattr(self._confirm_button, 'enabled', False)
        self._image_layer.changed.connect(disable_confirm)
        self._labels_layer.changed.connect(disable_confirm)
    
    def _create_anchors(self) -> Anchors:
        anchors = Anchors(
            name='Anchors', face_color='yellow', edge_color='black'
        )
        anchors.mode = 'select'
        anchors.metadata['grabber'] = self._grabber
        anchors.mouse_drag_callbacks.append(self._mouse_drag)
        anchors.events.contour.connect(self._on_contour_update)

        self._viewer.add_layer(anchors)
       
        return anchors
    
    def _on_contour_update(self, event=None) -> None:
        self._contour_layer.data = self._grabber.contour

    def _mouse_drag(self, anchors: Anchors, event) -> None:
        if self._grabber is None:
            return

        if anchors.mode == 'select':
            if len(anchors.selected_data) > 1:
                return
            # mouse press
            coords, _ = self._find_nearest_anchor(event.position)
            self._grabber.select(coords)
            yield
            # mouse move
            while event.type == 'mouse_move' and len(anchors.selected_data) == 1:
                coords = round(anchors.position[-2]), round(anchors.position[-1])
                if anchors.is_valid(coords) and not anchors.is_on_contour(coords):
                    self._grabber.drag(coords)
                    self._contour_layer.data = self._grabber.contour

                yield
            # mouse release
            self._grabber.confirm()
            anchors.selected_data = set()
    
    def _update_sigma(self, value: float) -> None:
        if self._grabber is None:
            return
        self._grabber.sigma = value

    def _update_epsilon(self, value: float) -> None:
        if self._grabber is None:
            return
        self._grabber.epsilon = value
        self._reset_anchors()
    
    def _reset_anchors(self) -> None:
        if self._grabber is None:
            return
        self._anchors_layer.data = np.array([p.coords for p in self._grabber.paths])
        self._anchors_layer.selected_data = set()
    
    def _create_mask(self) -> Optional[ArrayLike]: 
        labels = self._labels_layer.value.data
        if labels is None:
            return None
        
        if labels.ndim != 2:
            raise RuntimeError(f'Grabber only support 2-D labels, f{self.labels.ndim}-D found.')

        mask = labels == self._label_selection.value
        if mask.sum() == 0:
            warnings.warn(f'Selected label ({self._label_selection.value}) not found in `Labels`.')
            return None

        return mask.astype(np.uint8)

    def _validate_load(self) -> bool:
        return (self._image_layer.value is not None and
                self._labels_layer.value is not None)

    def _on_load(self) -> None:
        if self._labels_layer.value == self._contour_layer:
            warnings.warn('Input `Labels` cannot be `Contour.')
            return

        mask = self._create_mask()
        if mask is None:
            return
        image = self._image_layer.value.data

        if mask.shape != image.shape[:2]:
            raise RuntimeError(f'`Labels` and `Image` shape must match. Found {mask.shape} and {image.shape[:2]}')

        if image.ndim > 3:
            raise RuntimeError(f'`Image` must be 2 or 3-dimensional.')
        
        if self._preprocess.value:

            if image.ndim == 3 and image.shape[2] == 3:
                print('Converting RGB image to LAB ...')
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            image = cv2.bilateralFilter(image, 7, 25, -1)

        self._grabber = Grabber(
            image, mask, sigma=self._sigma.value, epsilon=self._epsilon.value
        )
        self._anchors_layer.metadata['grabber'] = self._grabber
        self._reset_anchors()
        self._contour_layer.data = self._grabber.contour
        self._confirm_button.enabled = True
        self._viewer.layers.selection = [self._anchors_layer]
    
    def _on_confirm(self) -> None:
        mask = self._grabber.mask
        labels = self._labels_layer.value.data
        selected = self._label_selection.value
        labels[labels == selected] = 0
        labels[mask] = selected
        self._labels_layer.value.data = labels
    
    def _find_nearest_anchor(self, coords: ArrayLike) -> Tuple[Tuple[int, int], float]:
        if self._grabber is None:
            raise RuntimeError('Grabber must be loaded before finding nearest anchor.')
        assert len(coords) == 2
        nearest = None
        min_dist = 1e23
        for p in self._grabber.paths:
            dist = (coords[0] - p.coords[0]) ** 2 + (coords[1] - p.coords[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = p.coords
        return nearest, np.sqrt(min_dist)
