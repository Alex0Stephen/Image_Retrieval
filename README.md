# 一、操作流程

1.图像数据库准备

将作为图库内容的图像放在目录 `./database/image`下，按以下语句运行，会自动读取所有图库图像、提取特征并保存至`./database/feature`下

>>> python run SaveFeature.py

2.开启检索系统

运行以下语句，可在本地显示系统操作界面

>>> python run SearchImage.py

# 二、结果展示

![image_result1](https://github.com/Alex0Stephen/Image_Retrival/master/image_result1.png)

![image_result2](C:\Users\AlexStephen\Desktop\北航\计算机视觉\Image_Retrival\image_result2.png)

![image_result3](C:\Users\AlexStephen\Desktop\北航\计算机视觉\Image_Retrival\image_result3.png)

# 三、requirement

torch == 1.8.0+cu111

torchvision == 0.9.0+cu111

pathlib == 2.3.5

numpy ==  1.21.5

matplotlib == 3.5.1

streamlit ==  1.8.1

urllib == 1.22

