# Image-processing and Computer version
	Review of what I learnt on image prcessing course.
	记录我对于ELEC4630 image processing 知识的掌握
## 1.Fundamentals
在图像分析中主要从4个方面分析
* 图像表示、编码及传输
* 图像增强
* 图像变更
* 图像分析（computer vision, image segmentation and deep learning）

如果将图像转为坐标中的坐标
![image](https://github.com/liuyu111/readme_add_pic/blob/master/directory/W1(1).PNG)
因此可以看出来当图像为0~255的灰度图时，XY坐标为图像的大小，Z坐标为图像的灰度。
对于图像格式中分为三种
* 2值图：图像中只有0或1
* 灰度图：图像为8bits 0~255
* 彩色图：图像为三通道8bits 也就是3*（0~255）

此外，图像的格式除了RGB，还有HSV等，主要受到颜色鲜艳程度以及光照等因素的干扰。因此受环境的影响比较大。
* RGB(Cartesian coordinate)
* YIQ or YUV(Cartesian coordinate 受色调和饱和度影响)
* HSV or HSI (与YUV相近 polar coordinates. – H denotes hue, S denotes saturation, V (I) denotes value (intensity))： 对于曝光严重的图像HSV的处理方法要好过RGB。
* CYMK (不常用)  

JPEG图像的加解码使用Discrete Cosine Transform(DCT)  
DCT-based encoder processing step(Source image data->DCT-based encoder->compressed image data)  
DCT-based decoder processing step(compressed image data->DCT-based decoder->Source image data)

## 2.Pyramids and Hough Transform（金字塔和霍夫变换）
### 边缘算子
对于图像分析，第一步应该就是图像分割，而图像分割与图像的边界相关，一下是几种边缘算子  
* Gradient: <a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;f&space;=&space;\left&space;\lfloor&space;\frac{delta_f}{delta_x},\frac{delta_f}{delta_y}&space;\right&space;\rfloor" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;f&space;=&space;\left&space;\lfloor&space;\frac{delta_f}{delta_x},\frac{delta_f}{delta_y}&space;\right&space;\rfloor" title="\Delta f = \left \lfloor \frac{delta_f}{delta_x},\frac{delta_f}{delta_y} \right \rfloor" /></a>
* Robert(2*2)
* Sobel(3*3)
* Sobel(5*5)
从最上面的算子Good localization, noise sensitive and poor detection --> poor localization, less noise sensitive and good detection. 

### 图像金字塔
Gaussian pyramid(低通)：1）使用filter对图像进行平滑处理 2）下采样 3）循环1&2操作得到更小，平滑度更高，分辨率更低的图像。
Laplacian pyramids(带通)：可以认为是残差金字塔，用来存储下采样后图片与原始图片的差异。因此拉普拉斯金字塔需要结合高斯金字塔使用。L_i = G_i - expand(G_(i+1))
应用：cv2.pyrDown;cv2.pyrUp


