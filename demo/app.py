import sys
import napari
import numpy as np
import cv2
from magicgui import magicgui
from qtpy.QtWidgets import QDoubleSpinBox

sys.path.insert(0, '.')
from src.grabber import Grabber
from src.mypoints import add_my_points


def main(args):

    image = cv2.cvtColor(cv2.imread(args[1]), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(args[2])
    mask = (mask.max(axis=2) > mask.mean()).astype(np.uint8)

    default_sigma = 30.0
    points_size = 5
    epsilon = 25

    lab_im = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_im = cv2.bilateralFilter(lab_im, 7, 25, -1)
    grabber = Grabber(image=lab_im,
                      mask=mask, sigma=default_sigma, epsilon=epsilon)

    with napari.gui_qt():
        viewer = napari.view_image(image)

        label = viewer.add_labels(grabber.contour,
                                  color={1: 'cyan'},
                                  name='contour', opacity=1.0)

        points = add_my_points(viewer, grabber, np.array([p.coords for p in grabber.paths]),
                               size=points_size, name='anchors', face_color='yellow', edge_color='black')
        points.mode = 'select'

        def find_closest(coords):
            minimum = None
            min_dist = 1e23
            for p in grabber.paths:
                dist = (coords[0] - p.coords[0]) ** 2 + (coords[1] - p.coords[1]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    minimum = p.coords
            return minimum, np.sqrt(min_dist)

        @points.mouse_drag_callbacks.append
        def mouse_click(layer, event):
            if layer.mode == 'select':
                if len(layer.selected_data) > 1:
                    return
                # mouse press
                coords, _ = find_closest(layer.coordinates)
                grabber.select(coords)
                yield
                # mouse move
                while event.type == 'mouse_move' and len(layer.selected_data) == 1:
                    coords = round(layer.position[0]), round(layer.position[1])
                    grabber.drag(coords)
                    label.data = grabber.contour
                    yield
                # mouse release
                grabber.confirm()
                layer.selected_data = set()

        @points.bind_key('Backspace')
        @points.bind_key('Delete')
        def remove_points(layer):
            for i in sorted(list(layer.selected_data), reverse=True):
                pt = grabber.paths[i].coords
                grabber.remove(pt)
            label.data = grabber.contour
            layer.remove_selected()

        @magicgui(auto_call=True,
                  sigma={'widget_type': QDoubleSpinBox, 'maximum': 255, 'minimum': 0.01, 'singleStep': 5.0})
        def update_sigma(sigma: float = default_sigma):
            grabber.sigma = sigma

        sigma_box = update_sigma.Gui()
        viewer.window.add_dock_widget(sigma_box, area='left')
        viewer.layers.events.changed.connect(lambda x: sigma_box.refresh_choices())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("app.py <input image path> <input mask path>")
        sys.exit(-1)
    main(sys.argv)