#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]  ## 读取的张量名称，注意和保存的节点相对应
pb_file         = "./yolov3_coco.pb"
image_path      = "./docs/images/road.jpeg"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()
print(tf.test.is_gpu_available())
original_image = cv2.imread(image_path)

original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])  ## float32
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name)

with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

# print(pred_sbbox.shape):(1,52,52,3,85)
pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

# print(pred_bbox.shape):(10647,85)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
# print(bboxes.shape):(113,6)
bboxes = utils.nms(bboxes, 0.45, method='nms')
# print(bboxes.shape):(30,6)
image = utils.draw_bbox(original_image, bboxes)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("result",image)


cv2.waitKey(0)
cv2.destroyAllWindows()