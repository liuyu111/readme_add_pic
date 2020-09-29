# Image-processing and Computer version
	Review of what I learnt on image prcessing course.
	记录我对于ELEC4630 image processing 知识的掌握
## Fundamentals
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

## 
