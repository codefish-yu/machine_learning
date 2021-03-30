'''
彩色图像的数组第三维顺序是bgr
'''

import cv2 as cv
import numpy as np

#打开图片
original = cv.imread('../课件/ml_data/forest.jpg')

print(original.shape,original[0,0])
#显示图片，这个方法是不阻塞的,第一个参数是名字
cv.imshow('origjnal',original)
#需要用waitkey阻塞，这个方法就是等待你按键盘，就不阻塞了(按任意键继续)


#切出某通道的数据
blue  = np.zeros_like(original)
blue[:,:,0] = original[:,:,0]   #将打开的original图片的蓝色通道(第三唯的数据)赋给blue图像
cv.imshow('blue',blue)


#生成绿色通道和蓝色通道同理
#直接对blue通道数组　*2可以加强blue的饱和度

#图像裁剪
h,w = original.shape[:2]
#算出底部和顶部
t,b = int(h/4),int(h*3/4)
#算出左右
l,r = int(w/4),int(w*3/4)
#切图
cropped = original[t:b,l:r] #t:b表示切第一维行，l:r表示切第二维列
cv.imshow('cropped',cropped)

#图像缩放
#第二个参数为缩放后的行列像素大小
resize1 = cv.resize(original,(150,100))
cv.imshow('resize1',resize1)

#第二个元素给None,第三第四个参数表示x,y轴变为原来的四倍
resize2 = cv.resize(resize1,None,fx=4,fy=4)

#图像保存
cv.imwrite('resize1.jpg',resize1)

cv.waitKey()

