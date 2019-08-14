from PIL import Image
import numpy
im = Image.open('manthanLogo.png','r').convert('RGBA')
np_im = numpy.array(im)
w,h,_ = np_im.shape
for i in range(w):
  for j in range(h):
    R,G,B,A = np_im[i][j]
    if(numpy.linalg.norm([R,G,B])>400):
      np_im[i][j] = [R,G,B,0]
    # if(np_im[i][j][0]>180):
      # np_im[i][j] = [0, 0]
    # if (np_im[i][j][0]<180 and np_im[i][j][0]>100):
      # np_im[i][j] = [np_im[i][j][0]-50, np_im[i][j][1]-80]

im_convert = Image.fromarray(np_im)
im_convert.save('shadow2.png')