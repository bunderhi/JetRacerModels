
from __future__ import print_function

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import sys, os
import time
import common
sys.path.insert(1, os.path.join(sys.path[0], ".."))

onnx_file_path = '/models/run10/jetracer.onnx'
engine_file_path = '/models/run10/jetracer.trt'
in_folder = '/models/train_data/Images'

i = 0
for filename in os.listdir(in_folder):
    i=i+1
    if i == 2:
        t0 = time.time()
    image = cv2.imread(os.path.join(in_folder,filename)).transpose(2,0,1).reshape(1,3,320,640)
    input = np.array(image, dtype=np.float32, order='C')/255
    output = np.ones((204800))
    flat = (output > 0.4).astype(np.uint8)
    im3 = flat.reshape (320,640)
    print('np sum',np.sum(flat))
t1 = time.time()
baselap = t1 - t0
i = i - 2
fps = i/baselap
print("Base: ",i,baselap,fps)


# Do inference with TensorRT
trt_outputs = []
with common.get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Do inference
    print('Running inference')
    
    i = 0
    for filename in os.listdir(in_folder):
        i=i+1
        if i == 2:
            t0 = time.time()
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        image = cv2.imread(os.path.join(in_folder,filename)).transpose(2,0,1).reshape(1,3,320,640)
        inputs[0].host = np.array(image, dtype=np.float32, order='C')/255
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        flat = (trt_outputs[0] > 0.4).astype(np.uint8)
        im3 = flat.reshape (320,640)
        print('np sum',np.sum(flat))
    t1 = time.time()
    lap = t1 - t0 - baselap
    i = i - 2
    fps = i/lap
    print("Inference: ",i,lap,fps)

