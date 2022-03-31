# Python Utilities

Python toolbox based on Python 3, PyTorch and OpenCV-Python.

:notebook: [[Wiki]](https://github.com/ryanxingql/pythonutils/wiki)

:e-mail: Feel free to contact: `ryanxingql@gmail.com`.

## 0. Archive

- v2: add new functions; improve efficiency; not compatible with v1.
- [v1](https://github.com/ryanxingql/pythonutils/tree/2d339029cd97f2a4acba288869bcc13f7daaf7de): the first formal version.

## 1. Content

- **algorithm**: a basis class of deep-learning algorithms.
- **conversion**: common-used tools for format/color space conversion, e.g., YCbCr <-> RGB conversion.
- **dataset**: basic datasets and common-used functions, e.g., LMDB dataset.
- **deep learning**: common-used tools for deep learning, e.g., multi-processing.
- **metrics**: common-used metrics, e.g., PSNR.
- **network**: a basis class of deep-learning networks.
- **system**: common-used tools for system manipulation, e.g., return formatted time data.

Besides, there are some other useful tools in `/individual/` that are not included in `/__init__.py`.

- `check_fetch_latest_model.py`: check and fetch latest model from another machine by SSH.
- `crop_two_images.py`: crop two images together according to the given axis.
- `fetch_remote_tf.py`: fetch remote log and tensorboard files.
- `occupy_gpu.py`: occupy GPU memory and/or utility of one GPU.

## 2. Principle

- In default, an image is a NUMPY array with the size being `(H W C)` and the data type being `np.uint8`.
- In default, an image input is first copied then processed; a tensor input is `[.cpu()].detach().clone()` then processed.

## 3. License

We adopt Apache License v2.0. For other licenses, please refer to [BasicSR](https://github.com/xinntao/BasicSR/tree/master/LICENSE).

If you find this repository helpful, you may cite:

```tex
@misc{2021xing,
  author = {Qunliang Xing},
  title = {Python Utils},
  howpublished = "\url{https://github.com/ryanxingql/pythonutils}",
  year = {2021},
  note = "[Online; accessed 11-April-2021]"
}
```
