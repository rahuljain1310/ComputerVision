import numpy
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def getBlackWhiteImage(fname,thresh):
  im = cv2.imread(fname)
  grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  _, imBw = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)
  return imBw

def getGaussianImage(fname,thresh,Gmatrix):
  im = cv2.imread(fname)
  grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  _, imBw = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)
  dst = cv2.GaussianBlur(imBw,Gmatrix,cv2.BORDER_DEFAULT)
  return dst

def getoverlay(thresh,imshape):
  w,h = imshape
  overlay = np.random.rand(w,h)*255
  return np.vectorize(lambda x: 255.0 if x > thresh else 0.0)(overlay)

def addNoise(img,sparsity,threshOverlay):
  overlay = getoverlay(threshOverlay, img.shape)
  noised = np.vectorize(lambda x: 255.0 if random.randint(1,255) > sparsity*255+(1-sparsity)*x else 0.0)(img)
  Noiseimg = 255.0-(overlay+noised)
  return Noiseimg

def getNoiseImage(fname, Gmatrix, Sparsity, threshBinary=127, threshOverlay=247.0):
  imBw = getBlackWhiteImage(fname,threshBinary)
  dst = cv2.GaussianBlur(imBw,Gmatrix,cv2.BORDER_DEFAULT)
  NoiseImg = addNoise(dst,Sparsity,threshOverlay)
  return NoiseImg

def addNoiseImage(image, Gmatrix, Sparsity, threshBinary=127, threshOverlay=247.0):
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, imBw = cv2.threshold(grayImage, threshBinary, 255, cv2.THRESH_BINARY)
  dst = cv2.GaussianBlur(imBw,Gmatrix,cv2.BORDER_DEFAULT)
  Noiseimg = addNoise(dst,Sparsity,threshOverlay)
  return Noiseimg

def writeImageToVideo(video,fr):
  # cv2.imshow('fr',fr)
  # cv2.waitKey(30)
  fr = np.asarray(fr, dtype=np.uint8)
  # imgWrite = cv2.cvtColor(fr,cv2.COLOR_GRAY2RGB)
  video.write(fr)

def writeMixture(img1,img2,video,N,Sparsity,threshOverlay):
  f = lambda x,y: math.cos(math.pi*(0.5*x)/y)
  for i in range(0,N):
    frame = addNoise(img1,Sparsity,threshOverlay=threshOverlay)
    writeImageToVideo(video,frame)
  for i in np.arange(0,N):
    a = f(i,N)
    img = a*img1+(1-a)*img2
    frame = addNoise(img,Sparsity,threshOverlay=threshOverlay)
    writeImageToVideo(video,frame)
  for i in range(0,int(N/2)):
    frame = addNoise(img2,Sparsity,threshOverlay=threshOverlay)
    writeImageToVideo(video,frame)

Sparsity = 0.7
Gmatrix = (7,7)
threshBinary = 127.0
threshOverlay = 254.4

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (5,150)
fontScale = 2
fontColor = (0,0,0)
lineType = 4

ImageList = ['Hey.png','Learn.png','Grow.png','Enrich.png','How.png','Come.png','ToMilan.png','Daily.png']
N = len(ImageList)

VideoPath = 'NoiseMerge.mp4'
img1 = getGaussianImage(ImageList[0],threshBinary,Gmatrix)
h,w = img1.shape
video = cv2.VideoWriter(VideoPath,-1,30,(w,h),isColor=False)

for k in range(1,N):
  print("Percentage Completed: {0: .2f} %".format((k-1)*100.0/N) )
  img2 = getGaussianImage(ImageList[k],threshBinary,Gmatrix)
  writeMixture(img1,img2,video,30,Sparsity,threshOverlay)
  img1 = img2

print("Percentage Completed: 100 %")