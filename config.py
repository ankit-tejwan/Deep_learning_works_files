
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.50#0.33
        self.width = 0.65#0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Correct path construction using os.path.join
        self.data_dir = os.path.join(os.getcwd(), "datasets", "images")
        self.train_ann = "train.json"
        self.val_ann = "valid.json"

        self.num_classes = 2
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1
        
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.1
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 10
        
        self.input_size = (640, 640)
        self.mosaic_scale = (0.5, 1.5)
        #self.random_size = (10, 20)
        self.test_size = (640, 640) #(960, 960)

