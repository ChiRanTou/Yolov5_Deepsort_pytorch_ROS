#!/usr/bin/env python3

import torch
import sys, os
import os
from torch.autograd import Variable
currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.datasets import letterbox

import rospy
import numpy as np
from sensor_msgs.msg import Image
from deepsort_yolov5.msg import BoundingBox, BoundingBoxes

from cv_bridge import CvBridge, CvBridgeError

import cv2

from rospkg import RosPack
package = RosPack()
package_path = package.get_path('deepsort_yolov5')

class YoloDetector():
    def __init__(self):
        rospy.loginfo("Initialize YOLO-V5 Detector")

        weights_name = rospy.get_param('~weights_name', 'yolov5/weights/yolov5s.pt')
        self.weights_path = os.path.join(package_path, 'scripts', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # topics
        self.image_topic = rospy.get_param('~image_topic', '/image_raw')
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic')
        self.published_image_topic = rospy.get_param('~detections_image_topic')
        # some parameters
        self.confidence_th = rospy.get_param('~confidence', 0.25)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        self.img_size = rospy.get_param('~img_size', 640)
        self.publish_image = rospy.get_param('~publish_image')
        self.gpu_id = rospy.get_param('~gpu_id', 0)

        # Initialize
        self.device = select_device(str(self.gpu_id))
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load YOLO-V5
        # self.detector = attempt_load(self.weights_path, map_location=self.device)  # load to FP32
        self.detector = torch.load(self.weights_path, map_location=self.device)['model'].float()  # load to FP32
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half()  # to FP16

        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names
        self.class_name = rospy.get_param('~class_name', None)
        if self.class_name == "None":
            self.class_name = None
        elif type(self.class_name) == str:
            self.class_name = self.names.index(self.class_name)
        else:
            self.class_name = self.class_name.split(', ')
            self.class_name = [self.names.index(int(i)) for i in self.class_name]

        # Initialize width and height
        self.h = 0
        self.w = 0

        self.classes_colors = {}        

        self.bridge = CvBridge()        

        # Subscribe to image topic
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)
        
        # Define publishers
        self.pub_ = rospy.Publisher(self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

        rospy.spin()
    
    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError as e:
            print(e)

        detection_results = BoundingBoxes()
        detection_results.header = msg.header
        detection_results.image_header = msg.header
        input_img = self.imagePreProcessing(self.cv_image)

        with torch.no_grad():
            detections = self.detector(input_img, augment=False)[0]  # list: bz * [ (#obj, 6)]
            detections = non_max_suppression(detections, self.confidence_th, self.iou_thres, classes=self.class_name, agnostic=False)

        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * (self.img_size / max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * (self.img_size / max(self.h, self.w))
                unpad_h = self.img_size - pad_y
                unpad_w = self.img_size - pad_x
                xmin_unpad = ((xmin-pad_x // 2) / unpad_w) * self.w
                xmax_unpad = ((xmax-xmin) / unpad_w) * self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y // 2)/unpad_h) * self.h
                ymax_unpad = ((ymax-ymin) / unpad_h) * self.h + ymin_unpad

                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = int(xmin_unpad.item())
                detection_msg.xmax = int(xmax_unpad.item())
                detection_msg.ymin = int(ymin_unpad.item())
                detection_msg.ymax = int(ymax_unpad.item())
                detection_msg.probability = conf.item()
                detection_msg.Class = self.names[int(det_class.item())]
                # print(detection_msg.xmin, detection_msg.xmax, detection_msg.ymin, detection_msg.ymax, detection_msg.probability, detection_msg.Class)

                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)

        # Publish detection results
        self.pub_.publish(detection_results)

        # Visualize detection results
        if (self.publish_image):
            self.visualize(detection_results, self.cv_image)
        
        return True

    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape

        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width

            # Determine image to be used
            self.padded_image = np.zeros((max(self.h, self.w), max(self.h, self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w - self.h) // 2: self.h + (self.w - self.h) // 2, :, :] = img
        else:
            self.padded_image[:, (self.h - self.w) // 2: self.w + (self.h - self.w) // 2, :] = img

        # Resize and normalize
        input_img = letterbox(self.padded_image, new_shape=self.img_size)[0]

        # Convert
        input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        input_img = np.ascontiguousarray(input_img)

        # numpy to tensor
        input_img = torch.from_numpy(input_img).to(self.device)
        input_img = input_img.half() if self.half else input_img.float()  # uint8 to fp16/32
        input_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if input_img.ndimension() == 3:
            input_img = input_img.unsqueeze(0)

        return input_img

    def visualize(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].Class
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0, 255, 3)
                self.classes_colors[label] = color

            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (int(color[0]), int(color[1]), int(color[2])),
                          thickness)
            text = ('{:s}: {:.3f}').format(label, confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1 + 20)), font, fontScale, (255, 255, 255), thickness,
                        cv2.LINE_AA)
        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        self.pub_viz_.publish(image_msg)        

if __name__=="__main__":
    # Initialize node
    rospy.init_node("detector_node")

    # Define detector object
    dm = YoloDetector()