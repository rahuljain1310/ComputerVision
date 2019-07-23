from PIL import Image
import numpy
from math import floor
import matplotlib.pyplot as plt

def getIntensityCount(np_img):
  KCount = numpy.zeros(256)
  for row in np_img:
    for K in row:
      KCount[K]+=1
  return KCount

def getCummulativeIntensity(np_count):
  Cumm = numpy.zeros(256)
  Cumm[0] = np_count[0]
  for i in range(1,256):
    Cumm[i] = Cumm[i-1]+np_count[i]    
  return Cumm  

im = Image.open('lenna.png','r')
im_grey = im.convert('L')
np_im_grey = numpy.array(im_grey)
w,h = np_im_grey.shape

norm_Intensity = getIntensityCount(np_im_grey)/(w*h)
Cumm_norm_Intensity = getCummulativeIntensity(norm_Intensity)

ScaleK = lambda t: int(floor(255*t))
Trans_Intensity = numpy.vectorize(ScaleK)(Cumm_norm_Intensity)

transformK = lambda t: Trans_Intensity[t]
np_equiImg = numpy.vectorize(transformK)(np_im_grey)

equi_im = Image.fromarray(np_equiImg)

## Show Both Images
concat_np = numpy.concatenate((np_im_grey,np_equiImg))
concat_img = Image.fromarray(concat_np)
concat_img.show()

## Plot Cummulative Frequency Histograms
Trans_NormIntensity = getIntensityCount(np_equiImg)/(w*h)
Trans_CommIntensity = getCummulativeIntensity(Trans_NormIntensity)

plt.plot(Cumm_norm_Intensity)
plt.plot(Trans_CommIntensity)
plt.show()