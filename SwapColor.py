from PIL import Image
import numpy
from math import floor
import matplotlib.pyplot as plt

## RGB
red = (195,1,38)
blue = (101,126,148)
black = (1,13,35)
white = (239,213,180)

im = Image.open('ppEdit.png','r').convert('RGB')
np_im = numpy.array(im)
w,h,_ = np_im.shape
np_im_convert = numpy.zeros(np_im.shape,dtype='uint8')

for i in range(w):
  for j in range(h):
    pixel = np_im[i][j]
    if( numpy.linalg.norm(pixel-white) < 0.1 ):
      np_im_convert[i][j] = numpy.array(red)
      continue
    if( numpy.linalg.norm(pixel-red) < 0.1 ):
      np_im_convert[i][j] = numpy.array(white)
      continue
    np_im_convert[i][j] = pixel

im_convert = Image.fromarray(np_im_convert)
im_convert.save('ppEditSwapped.png')
