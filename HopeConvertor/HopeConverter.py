from PIL import Image
import numpy
from math import floor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations 

## RGB
red = (195,1,38,255)
blue = (101,126,148,255)
black = (1,13,35,255)
white = (239,213,180,255)

## Color Selection
color1 = red
color2 = blue
color3 =  black
color4 = white
color = numpy.array([color1,color2,color3,color4],dtype='uint8')

def closestColor(npt, color_clusters):
  disArr = list(map(lambda t: numpy.linalg.norm(t-npt) ,color_clusters))
  i = numpy.argmin(disArr)
  return color_clusters[i]

mode = 'RGBA'
pixelDim = len(mode)
imagename = input()
im = Image.open(imagename,'r').convert(mode)
np_im = numpy.array(im)
w,h,_ = np_im.shape
np_im_flatten = np_im.reshape((w*h,pixelDim))
np_im_png = np_im_flatten[numpy.linalg.norm(np_im_flatten) !=0][0]
print(np_im_png.shape)
kmeans = KMeans(n_clusters=4, random_state=0).fit(np_im_png)
color_clusters = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
np_im_convert = numpy.zeros(np_im.shape,dtype='uint8')

ClusteringMap = []
possibleMaps = list(permutations(color))
minerror = 100000
for c in possibleMaps:
  c = numpy.array(c)
  error = 0
  for i in range(0,4):
    error += numpy.linalg.norm(c[i]-color_clusters[i])
  if(error<minerror):
    minerror = error
    ClusteringMap = c

print(ClusteringMap)
for i in range(w):
  for j in range(h):
    ## Avoid Transparency
    # if(numpy.linalg.norm(np_im[i][j])==0 or np_im[i][j][3]==0 ):
      # continue
    x = kmeans.predict([np_im[i][j]])[0]
    np_im_convert[i][j]  = ClusteringMap[x]

concat_np_img = numpy.concatenate((np_im,np_im_convert))
im_convert = Image.fromarray(concat_np_img)
im_convert.save('ppEdit.png')
