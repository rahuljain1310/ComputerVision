from PIL import Image
import numpy
from math import floor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations 


red = (195,1,38)
blue = (101,126,148)
black = (1,13,35)
white = (239,213,180)
color = numpy.array([red,blue,black,white],dtype='uint8')


def closestColor(npt, color_clusters):
  disArr = list(map(lambda t: numpy.linalg.norm(t-npt) ,color_clusters))
  i = numpy.argmin(disArr)
  return color_clusters[i]

im = Image.open('lenna.png','r').convert('RGB')
np_im = numpy.array(im)
w,h,_ = np_im.shape
np_im_flatten = np_im.reshape((w*h,3))
kmeans = KMeans(n_clusters=4, random_state=0).fit(np_im_flatten)
color_clusters = kmeans.cluster_centers_
print(kmeans.cluster_centers_)
np_im_convert = numpy.zeros(np_im.shape,dtype='uint8')


ClusteringMap = []
possibleMaps = list(permutations(color_clusters))
minerror = 100000
for c in possibleMaps:
  c = numpy.array(c)
  error = 0
  for i in range(0,4):
    error += numpy.linalg.norm(c[i]-color[i])
  if(error<minerror):
    minerror = error
    ClusteringMap = c
  
# ClusteringMap = []
# for c in color:
#   col = closestColor(c,color_clusters)
#   ClusteringMap.append(col)

print(ClusteringMap)
for i in range(w):
  for j in range(h):
    x = kmeans.predict([np_im[i][j]])[0]
    np_im_convert[i][j]  = ClusteringMap[x]

concat_np_img = numpy.concatenate((np_im,np_im_convert))
im_convert = Image.fromarray(concat_np_img)
im_convert.show()
