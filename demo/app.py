import sys
import napari
import numpy as np
import cv2
from magicgui import magicgui

sys.path.insert(0, '.')
from src.grabber import Grabber
from src.mypoints import add_my_points


def main(args):

    image = cv2.cvtColor(cv2.imread(args[1]), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(args[2])
    mask = (mask.max(axis=2) > mask.mean()).astype(np.uint8)

    default_sigma = 30.0
    points_size = 10
    default_epsilon = 25

    lab_im = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_im = cv2.bilateralFilter(lab_im, 7, 25, -1)
    grabber = Grabber(image=lab_im,
                      mask=mask, sigma=default_sigma, epsilon=default_epsilon)

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
            coords, _ = find_closest(event.position)
            grabber.select(coords)
            yield
            # mouse move
            while event.type == 'mouse_move' and len(layer.selected_data) == 1:
                coords = round(layer.position[0]), round(layer.position[1])
                if layer.is_valid(coords, on_contour=False):
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

    @magicgui(
        auto_call=True,
        sigma=dict(widget_type='FloatSlider', max=255, min=0.01, step=5.0,
                   tooltip='Lower values makes contour adheres more to the image (and noise) boundaries.'),
        eps=dict(widget_type='FloatSlider', max=500, min=1, step=5,
                 tooltip='Smaller values estimates more anchor points'),
    )
    def update_params(sigma: float = default_sigma, eps: float = default_epsilon):
        if grabber.sigma != sigma:
            grabber.sigma = sigma
        elif eps != grabber.epsilon:
            grabber.epsilon = eps
            points.data = np.array([p.coords for p in grabber.paths])
        else:
            grabber.optimum_contour()
            label.data = grabber.contour
            grabber.recompute_anchors()
            points.data = np.array([p.coords for p in grabber.paths])
            
    viewer.window.add_dock_widget(update_params, area='right')
    viewer.layers.events.changed.connect(lambda x: params_box.refresh_choices())

    napari.run()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("app.py <input image path> <input mask path>")
        sys.exit(-1)
    main(sys.argv)
