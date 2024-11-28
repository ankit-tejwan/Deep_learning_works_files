
##############with distance validations infer######################
##import warnings
##warnings.filterwarnings("ignore")
##import time
##import os
##import torch
##import importlib
##import cv2 
##import numpy as np
##import pandas as pd
##import matplotlib.pyplot as plt
##from PIL import Image
##from IPython.display import display
##from yolox.utils import postprocess
##from yolox.data.data_augment import ValTransform
##from threading import Thread
##from queue import Queue
##
##
##
##CHECKPOINT_FILE = r"G:\YOLOX-main\YOLOX_training\YOLOX_outputs\config\epoch_99_ckpt.pth"
##COCO_CLASSES = ("warsal",)
##
### Get YOLOX experiment
##current_exp = importlib.import_module('config')
##exp = current_exp.Exp()
##
### Set inference parameters
##test_size = (640, 640)
##num_classes = 1
##confthre = 0.5
##nmsthre = 0.5
##
### Get YOLOX model
##model = exp.get_model()
##model.cuda()
##model.eval()
##
### Load custom trained checkpoint
##ckpt_file = CHECKPOINT_FILE
##ckpt = torch.load(ckpt_file, map_location="cpu")
##model.load_state_dict(ckpt["model"])
##
##def yolox_inference(img, model, test_size): 
##    bboxes = []
##    bbclasses = []
##    scores = []
##    
##    preproc = ValTransform(legacy=False)
##
##    tensor_img, _ = preproc(img, None, test_size)
##    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
##    tensor_img = tensor_img.float()
##    tensor_img = tensor_img.cuda()
##
##    with torch.no_grad():
##        outputs = model(tensor_img)
##        outputs = postprocess(
##                    outputs, num_classes, confthre,
##                    nmsthre, class_agnostic=True
##                )
##
##    if outputs[0] is None:
##        return [], [], []
##    
##    outputs = outputs[0].cpu()
##    bboxes = outputs[:, 0:4]
##    bboxes /= min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
##    bbclasses = outputs[:, 6]
##    scores = outputs[:, 4] * outputs[:, 5]
##    
##    return bboxes, bbclasses, scores
##
##def draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, classes_dict):
##    for i in range(len(bboxes)):
##        box = bboxes[i]
##        cls_id = int(bbclasses[i])
##        score = scores[i]
##        if score < confthre:
##            continue
##        x0 = int(box[0])
##        y0 = int(box[1])
##        x1 = int(box[2])
##        y1 = int(box[3])
##
##        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
##        cv2.putText(img, '{}:{:.1f}%'.format(classes_dict[cls_id], score * 100), (x0, y0 - 3), 
##                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=2)
##    return img
##
##def is_within_roi(bbox, roi):
##    x0, y0, x1, y1 = bbox
##    roi_x0, roi_y0, roi_x1, roi_y1 = roi
##    # Check if the bounding box is completely inside the ROI
##    return roi_x0 <= x0 <= roi_x1 and roi_y0 <= y0 <= roi_y1
##
##def track_detections_in_roi(bboxes, rois, roi_flags):
##    detection_status = {}
##    
##    for i, bbox in enumerate(bboxes):
##        for roi_index, roi in enumerate(rois):
##            if is_within_roi(bbox, roi):
##                # If object is inside ROI, flag it as True and assign ID
##                detection_status[i] = (roi_index, True)  # ROI index and status (inside ROI)
##                roi_flags[roi_index] = True  # Mark flag for this ROI ID as True
##                break
##        else:
##            detection_status[i] = (None, False)  # Not inside any ROI
##    return detection_status
##
##def result(img, output_queue, rois, roi_flags):
##    # Get predictions
##    img = cv2.resize(img, (640, 640))
##    bboxes, bbclasses, scores = yolox_inference(img, model, test_size)
##
##    # Track detections in ROIs
##    detection_status = track_detections_in_roi(bboxes, rois, roi_flags)
##
##    for i, (roi_index, inside_roi) in detection_status.items():
##        if inside_roi:
##            print(f"Object {i} detected inside ROI {roi_index}.")
##        else:
##            print(f"Object {i} is outside all ROIs.")
##
##    # Draw predictions on the image
##    out_image = draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, COCO_CLASSES)
##    
##    # Convert BGR to RGB for Matplotlib
##    out_image_rgb = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
##    output_queue.put(out_image_rgb)
##
##output_queue = Queue()
##processing_complete = True
##
### Define video source
##VIDEO_SOURCE = r"C:\Users\Admin\MVS\Data\MV-CS050-10GC (DA1898288)\Video_20241015144058361.avi"
##cap = cv2.VideoCapture(VIDEO_SOURCE)
##
### Define ROIs (list of tuples: (x0, y0, x1, y1))
##rois = [(100, 120, 585, 210), (15, 150, 85, 600)]
### Initialize flags for ROIs (False initially, will be True if object stays in ROI)
##roi_flags = [False] * len(rois)
##
### Loop to process video frames
##while cap.isOpened():
##    ret, img = cap.read()
##    if not ret:
##        break  # Break if video ends or error occurs
##    if processing_complete:
##        processing_complete = False
##        # Process the frame in a separate thread
##        thread = Thread(target=result, args=(img, output_queue, rois, roi_flags))
##        thread.start()
##
##    if not output_queue.empty():
##        framed = output_queue.get()
##        img_combined = cv2.resize(framed, (800, 800))
##        processing_complete = True
##        cv2.imshow("Video Stream", img_combined)
##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            print("roi_flags : ",roi_flags)
##            break
##print("roi_flags : ",roi_flags)
##cap.release()
##cv2.destroyAllWindows()



#########without distance detections infer#################
##
##import warnings
##warnings.filterwarnings("ignore")
##import time
##import os
##import torch
##import importlib
##import cv2 
##import pandas as pd
##import matplotlib.pyplot as plt
##from PIL import Image
##from IPython.display import display
##from yolox.utils import postprocess
##from yolox.data.data_augment import ValTransform
##from threading import Thread
##from queue import Queue
##
##CHECKPOINT_FILE=r"G:\YOLOX-main\YOLOX_training\YOLOX_outputs\config\epoch_99_ckpt.pth"
##COCO_CLASSES = (
##  "warsal",
##)
##
### get YOLOX experiment
##current_exp = importlib.import_module('config')
##exp = current_exp.Exp()
##
### set inference parameters
##test_size = (640, 640)
##num_classes = 1
##confthre = 0.5
##nmsthre = 0.4
##
##
### get YOLOX model
##model = exp.get_model()
##model.cuda()
##model.eval()
##
### get custom trained checkpoint
##ckpt_file = CHECKPOINT_FILE
##ckpt = torch.load(ckpt_file, map_location="cpu")
##model.load_state_dict(ckpt["model"])
##
##def yolox_inference(img, model, test_size): 
##    bboxes = []
##    bbclasses = []
##    scores = []
##    
##    preproc = ValTransform(legacy = False)
##
##    tensor_img, _ = preproc(img, None, test_size)
##    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
##    tensor_img = tensor_img.float()
##    tensor_img = tensor_img.cuda()
##
##    with torch.no_grad():
##        outputs = model(tensor_img)
##        outputs = postprocess(
##                    outputs, num_classes, confthre,
##                    nmsthre, class_agnostic=True
##                )
##
##    if outputs[0] is None:
##        return [], [], []
##    
##    outputs = outputs[0].cpu()
##    bboxes = outputs[:, 0:4]
##
##    bboxes /= min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
##    bbclasses = outputs[:, 6]
##    scores = outputs[:, 4] * outputs[:, 5]
##    
##    return bboxes, bbclasses, scores
##
##def draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, classes_dict):
##    for i in range(len(bboxes)):
##            box = bboxes[i]
##            cls_id = int(bbclasses[i])
##            score = scores[i]
##            if score < confthre:
##                continue
##            x0 = int(box[0])
##            y0 = int(box[1])
##            x1 = int(box[2])
##            y1 = int(box[3])
##
##            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
##            cv2.putText(img, '{}:{:.1f}%'.format(classes_dict[cls_id], score * 100), (x0, y0 - 3), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), thickness = 5)
##    return img
##
##def result(img,output_queue):
##    # Get predictions
##    bboxes, bbclasses, scores = yolox_inference(img, model, test_size)
##
##    # Draw predictions
##    out_image = draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, COCO_CLASSES)
##
##    # Convert BGR to RGB for Matplotlib
##    out_image_rgb = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
##    output_queue.put(out_image_rgb)
##    
##output_queue = Queue()
##processing_complete = True
### Define video source (0 for webcam or a video file path)
##VIDEO_SOURCE = r"C:\Users\Admin\MVS\Data\MV-CS050-10GC (DA1898288)\Video_20241015144058361.avi"
##cap = cv2.VideoCapture(VIDEO_SOURCE)
##
### Define parameters
##confthre = 0.5  # Confidence threshold
##test_size = (640, 640)  # Input size for your model
##COCO_CLASSES = ["warsal"]  # Define your classes list
##
### Loop to process video frames
##while cap.isOpened():
##    ret, img = cap.read()
##    if not ret:
##        break  # Break if the video ends or there's an error
##    if processing_complete:
##        processing_complete = False
##        # Process the frame in a separate thread
##        thread = Thread(target=result, args=(img,output_queue))
##        thread.start()
##        
##    if not output_queue.empty():
##        framed = output_queue.get()
##        img_combined = cv2.resize(framed, (800, 800))
##        processing_complete = True
##        cv2.imshow("Video Stream", img_combined)
##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            break
##
##cap.release()
##cv2.destroyAllWindows()
