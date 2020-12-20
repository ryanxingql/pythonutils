# Python Utils

Python is all we need!

Feel free to contact: ryanxingql@gmail.com.

## Content

- **algorithm**: basic template for deep learning algorithms.
- **conversion**: common-used tools for format/color space conversion, e.g., YCbCr <-> RGB conversion.
- **dataset**: basic datasets and common-used functions, e.g., LMDB dataset.
- **deep learning**: common-used tools for deep learning, e.g., multi-processing.
- **metrics**: common-used metrics, e.g., PSNR.
- **network**: basic template for deep learning networks.
- **system**: common-used tools for system manipulation, e.g., return formatted time data.

## Principle

- Image default format: (H W C) uint8 numpy array.
- In default, input is first copied then processed. Note that `np.copy` is a shallow copy and will not copy object elements within arrays. See [ref](https://numpy.org/doc/stable/reference/generated/numpy.copy.html).
- In default, input image is first converted to `np.float64`, then converted back after processing.
