
import torch
from torch import jit
import cv2
import time


# Load the torchscript model 
model = jit.load('/models/run10/model.zip')
# Set the model to evaluate mode
model = model.cuda().eval()


ino = 536
# Read  a sample image and mask from the data-set
img = cv2.imread(f'/models/train_data/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,640)
mask = cv2.imread(f'/models/train_data/Masks/{ino:03d}_mask.png')


torch.cuda.current_stream().synchronize()
t0 = time.time()

with torch.no_grad():
    a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
 
torch.cuda.current_stream().synchronize()
t1 = time.time()
lap = t1 - t0
print("Torch: ",lap)

pt_pred = a.cpu().detach().numpy()[0][0]>0.4
print("Torch Shape",pt_pred.shape)


