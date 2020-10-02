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

### pyramids(图像金字塔)
Gaussian pyramid(低通)：1）使用filter对图像进行平滑处理 2）下采样 3）循环1&2操作得到更小，平滑度更高，分辨率更低的图像。  
Laplacian pyramids(带通)：可以认为是残差金字塔，用来存储下采样后图片与原始图片的差异。因此拉普拉斯金字塔需要结合高斯金字塔使用。L_i = G_i - expand(G_(i+1))  
应用：cv2.pyrDown;cv2.pyrUp  

### Hough transform(霍夫变换)
通过将直线或圆的表达式转化到Hesse仿射坐标系（roi,theta），直线或圆交于仿射坐标系上的每一点为直线或圆的特征

## 3.Image Representations and Morphology（图像表示和形态学）
### Image representation(图像表示)
数字图像DFT
空间域和频域
* 空间域：在图像处理中，时域可以理解为空间域或者图像空间，处理对象为图像像元。
* 频域：以空间频域为自变量描述图像的特性。
空间域与频域可互相转换，对图像施行 **二维离散傅立叶变换**或**小波变换**，可以将图像由空间域转换到频域；通过对应的反变换又可转换回空间域图像，即人可以直接识别的图像。 
二维傅里叶变换采取换位方法使直流成分出现在中央（中心化），变换后中心为低频，外面为高频。
图像的锐化和模糊
* 截取频率的低频分量，对其作傅里叶变换，得到的就是模糊后的图像，即**低通滤波**  
* 截取频率的高频分量，对其作傅里叶变换，得到的就是锐化后的图像，即**高通滤波**  
傅里叶变化中的Magnitude在自然的图像中基本相同。
图像表示的几种方法  
* Chain codes: 基于4联通或8联通
* Signatures: 用1维数组来表示边界
### Thresholding(阈值)
* Automatic thresholding --Otsu's method MATAB中的grayscale
* Adaptive thresholding --1) adaptive mean threshold 2)adaptive gaussian threshold
### Morphology(形态学)
形态学技术被用来找边界，skeletons，convex hulls
腐蚀与膨胀；开运算与闭运算
结合图像的开闭运算，形态学又有thinning,thickening,skeletonization and pruning

## 4.Greyscale Morphology, Energy Minimization, 3D Reconstruction(灰度形态学，能量最小，3D重构)
### 灰度形态学运算
许多二进制形态学运算简单的扩展到灰度图像
因此就有了灰度图膨胀、腐蚀、开运算与闭运算  
分水岭算法
* 四叉树平滑（多分辨率操作，Pyramids）
* 对底层分类的水浸（将图片按照灰度向上延伸，水浸底层低灰度light pixels become flooded first）
* 重新边界预测（在新的分辨率下重复水浸，产生新的边界，直到图片达到原分辨率）

### Discrete energy minimisation
* Dynamic progamming/Viterbi
example: string matching,stero matching
* Dijkstra's Algorithm and fast marching
example: path planning
active contours(snake)
* 最初的snake模型基于能量最小化框架，但是不能处理变形过程中的**拓扑变化**，从而不能用于检测多目标的情况。
* 后来将曲线量化为level set函数（等势线），但是曲线容易越过边界运动到物体内部
* Geodesic active contours 使得曲线可以停在目标物体的边缘

### 3D Reconstruction
3D重建分为稀疏重构或者稠密重构
shape from silhouette
相机有两个最主要参数
* intrinsic calibration内在校准（内参）
* extrinsic callibration外在校准（外参）[X_w,Y_w,Z_w,W_w]
标定获得3维和2维关系
* step1: 计算外参转换
extrinsic matrix -> projective matirx  
**[X,Y,Z,W]**
* step2: 计算投影矩阵
3D projective matrix -> 2D projective matrix  
**[X_i,Y_i,W_i]**
* step3: 添加内参获得图像
P_i = A*P_i*V*v_m  
**[X_w,Y_w,Z_w,W_w]**

## 5.Projective Geometry （投影结构）
## 6.Biometrics and Face detection (生物识别和人脸识别)
* 指纹识别
* 人脸识别（最常见）
* 虹膜识别（精度最高）
* 声音识别（精度最低）
Receiver Operating Characteristic Curve(ROC)

## 8.Face recogntion and srveillance
## 9.Deep learning for computer vision
### CNN for image analysis
three main layer types
* input layer(输入层)-图像
* convolutional layer(卷积层)-可调滤波器
* pooling layer(池化层)-减少图像尺寸
* fully-connected layer(全连接层)-分类和做决策
图像分析的框架[input-connv-relu-connv-relu-pool-fc]
* input 原始图片的尺寸包含R,G,B三通道
* conv 根据神经元的输出确定连接Input的尺寸
* relu 将应用逐元素激活（高斯滤波）
* pool 下采样，减小图片尺寸
* FC 计算每个归类的分数然后进行分类

### object detection
对象检测是对图像的分类
* backbone detector
* Post CNN
* generative adversarial networks(gan)
DNN 框架
* google:tensorflow
* facebook: torch,pytorch,caffe2
* caffe,caffe2,keras,mxnet,darnet,matconvnet,matlab2017,etc

## 10.getting in deep(更深度学习)
安装 AI/DL 在conda环境中
