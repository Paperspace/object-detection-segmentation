import PIL
from PIL import Image


def _scale_to_square(orig, targ):
  # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
  # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
  targ_sz = (targ, targ)
  return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)


def _unsquare(image, orig):
  targ_sz = orig.size
  image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
  return image


