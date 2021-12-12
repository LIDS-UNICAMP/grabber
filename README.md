# Grabber: A Tool to Improve Convergence in Interactive Image Segmentation

[![License](https://img.shields.io/pypi/l/grabber.svg?color=green)](https://github.com/LIDS-UNICAMP/grabber/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/grabber.svg?color=green)](https://pypi.org/project/grabber)
[![Python Version](https://img.shields.io/pypi/pyversions/grabber.svg?color=green)](https://python.org)
[![tests](https://github.com/LIDS-UNICAMP/grabber/workflows/tests/badge.svg)](https://github.com/LIDS-UNICAMP/grabber/actions)
[![codecov](https://codecov.io/gh/LIDS-UNICAMP/grabber/branch/master/graph/badge.svg)](https://codecov.io/gh/LIDS-UNICAMP/grabber)

A tool for contour-based segmentation correction (2D only).

This repository provides a demo code of the paper:
> **Grabber: A Tool to Improve Convergence in Interactive Image Segmentation**
> [Jordão Bragantini](https://jookuma.github.io/), Bruno Moura, [Alexandre X. Falcão](http://lids.ic.unicamp.br/), [Fábio A. M. Cappabianco](https://scholar.google.com/citations?user=qmH9VEEAAAAJ&hl=en&oi=ao)

https://user-images.githubusercontent.com/21022743/145699960-57da06a5-668f-4e81-82b5-7f3d3ddf8ee3.mp4

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `grabber` via [pip]:

    pip install grabber


## Known Limitations

This implementation doesn't support the items below, feel free to open a PR to add them.

- It only support 2D image, supporting 3D images isn't trivial, but it could be applied per slice with minor changes.

## Citation

If this work was useful for your research, please cite our paper:

```
@article{bragantini2020grabber,
  title={Grabber: A Tool to Improve Convergence in Interactive Image Segmentation,
  author={Bragantini, Jord{\~a}o and Bruno Moura, Falc{\~a}o, Alexandre Xavier and Cappabianco, F{\'a}bio AM,
  journal={Pattern Recognition Letters},
  year={2020}
}
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"grabber" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
