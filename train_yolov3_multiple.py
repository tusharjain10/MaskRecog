# Check if NVIDIA GPU is enabled
!nvidia-smi

from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

# Clone
!git clone https://github.com/AlexeyAB/darknet

# Configure
# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

# Compile
!make

# Make a copy of yolov3.cfg
!cp cfg/yolov3.cfg cfg/yolov3_training.cfg

# Change lines in yolov3.cfg file
!sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
!sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
!sed -i 's/max_batches = 500200/max_batches = 400/' cfg/yolov3_training.cfg
!sed -i '610 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '696 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '783 s@classes=80@classes=2@' cfg/yolov3_training.cfg
!sed -i '603 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '689 s@filters=255@filters=21@' cfg/yolov3_training.cfg
!sed -i '776 s@filters=255@filters=21@' cfg/yolov3_training.cfg

!echo -e 'mask\no mask' > data/obj.names
!echo -e 'classes= 2\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /mydrive/yolov3' > data/obj.data

!cp cfg/yolov3_training.cfg /mydrive/yolo/yolov3_testing.cfg
!cp data/obj.names /mydrive/yolo/classes.txt

!mkdir data/obj
!unzip /mydrive/yolo/images.zip -d data/obj

import glob
images_list = glob.glob("data/obj/*.jpg")
with open("data/train.txt", "w") as f:
    f.write("\n".join(images_list))

!wget https://pjreddie.com/media/files/darknet53.conv.74

!./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show

# Uncomment below and comment above to re-start your training from last saved weights
#!./darknet detector train data/obj.data cfg/yolov3_training.cfg /mydrive/yolov3/yolov3_training_last.weights -dont_show