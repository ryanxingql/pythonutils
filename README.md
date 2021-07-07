# Python Utilities

Utilities based on Python 3, PyTorch and OpenCV-Python.

:notebook: [[Wiki]](https://github.com/RyanXingQL/PythonUtils/wiki)

:e-mail: Feel free to contact: `ryanxingql@gmail.com`.

## 0. Archive

- v2
- [v1](https://github.com/RyanXingQL/PythonUtils/tree/2d339029cd97f2a4acba288869bcc13f7daaf7de): the first formal version.

## 1. Content

- **algorithm**: a basis class of deep-learning algorithms.
- **conversion**: common-used tools for format/color space conversion, e.g., YCbCr <-> RGB conversion.
- **dataset**: basic datasets and common-used functions, e.g., LMDB dataset.
- **deep learning**: common-used tools for deep learning, e.g., multi-processing.
- **metrics**: common-used metrics, e.g., PSNR.
- **network**: a basis class of deep-learning networks.
- **system**: common-used tools for system manipulation, e.g., return formatted time data.

Besides, there are some other common-used tools in `individual.py` that are not included in this package.

## 2. Principle

- In default, an image is a NUMPY array with the size being `(H W C)` and the data type being `np.uint8`.
- In default, an image input is first copied then processed; a tensor input is `[.cpu()].detach().clone()` then processed.

## 3. License

If you find this repository helpful, you may cite:

```tex
@misc{PythonUtils_xing_2021,
  author = {Qunliang Xing},
  title = {Python Utils},
  howpublished = "\url{https://github.com/RyanXingQL/PythonUtils}",
  year = {2021},
  note = "[Online; accessed 11-April-2021]"
}
```
