# YOLOV5 + Deepsort Tracking with ROS Noetic and Pytorch
---

## Prepare
1. **Install ROS Noetic**

2. **Create a virtual environment with Python >= 3.8**

```bash
    conda create -n yolov5_deepsort python=3.8
    conda activate yolov5_deepsort
```

2. **Install pytorch and torchvision**

```bash
    pip3 install torch torchvision torchaudio
```
you can follow the official website to install the correct version of [pytorch](https://pytorch.org/get-started/locally/).

*cuda **12.2**, cudnn **8.9.7.29** is tested in this branch. You can try other versions, but I am not sure if it will work.*

3. **Install Python3 Dependencies**

```bash
    pip3 install rospkg catkin_pkg
```

4. **Clone the repository**

- Clone the repository to your catkin workspace
```bash
    mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
    git clone https://github.com/ChiRanTou/Yolov5_Deepsort_pytorch_ROS.git
    cd ..
    catkin_make
```

- Ensure that all dependencies are met.
```bash
    cd ~/catkin_ws/src/Yolov5_Deepsort_pytorch_ROS
    pip3 install -r requirements.txt
```

- You can download YOLOv5 weight from [the official website of yolov5](https://github.com/ultralytics/yolov5) and place the downloaded `.pt` file under `yolov5/weights/`. **I've already put the `yolov5s.pt` in the folder. You can other weight file if you like.**

- You may also need to download the deepsort weight file from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) and place `ckpt.t7` file under `deep_sort/deep/checkpoint/`. **I've also already put the `ckpt.t7` in the folder. You can the file if you like.**

---
## Run

***Before running the Project, you may notice that a ROS simulation enviornment is required. A robot with rgb camera is also needed to send the sensor_msgs/Image topic. So you have to get one first.***

1. open the launch file and change the `image_topic` to the topic that your camera publish the image.

```xml
    <arg name="image_topic" default="/rgb/image_raw"/>
```

2. start your ROS simulation enviornment and make sure the camera is working.

3. launch the task you want to run.

```bash
    # for dectection only
    roslaunch yolov5_deepsort detector.launch

    # for tracking
    roslaunch yolov5_deepsort tracker.launch
```

4. open ``rviz`` if you didn't open it, and add the `detected_objects_image/IMAGE` or `tracked_objects_image/IMAGE` based on your task to the display panel. You can now see the result in the rviz.

---

## Reference
- [yolov5_deepsort_ros](https://github.com/Jungsu-Yun/yolov5_deepsort_ros)
- [DeepSORT_YOLOv5_Pytorch](https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch)
- [yolov5](https://github.com/ultralytics/yolov5)
- [deep_sort](https://github.com/nwojke/deep_sort)

Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 