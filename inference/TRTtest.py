import torch
import torch2trt
import time
import cv2

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']

def equalize(im):
    gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cli = clahe.apply(gray)
    im2 = cv2.cvtColor(cli,cv2.COLOR_GRAY2RGB)
    return im2

model = torch.load('/home/brian/models/run09/weights.pt')
model = model.cuda().eval().half()
model_w = ModelWrapper(model).half()
data = torch.ones((1,3,320,640)).cuda().half()
print('start ...')
t0 = time.time()
model_trt = torch2trt.torch2trt(model_w, [data], max_workspace_size=1 << 20, fp16_mode=True)
t1 = time.time()
print('finish ',t1-t0)
#torch.save(model_trt.state_dict(), '/home/brian/models/jetson/model_trt.pt')
ino = 858
# Read  a sample image
img = cv2.imread(f'/home/brian/train_data//{ino:03d}.jpg')
#img2 = equalize(img)
#crop_img = img2[320:550, 200:700].copy()
#img3 = crop_img.transpose(2,0,1).reshape(1,3,230,500)
input = torch.from_numpy(img).type(torch.cuda.HalfTensor)/255

print('tensorRT test')
torch.cuda.current_stream().synchronize()
t0 = time.time()
output = model_trt(input)
torch.cuda.current_stream().synchronize()
t1 = time.time()
print(1.0 / (t1 - t0))

torch.cuda.current_stream().synchronize()
t0 = time.time()
output = model_trt(input)
torch.cuda.current_stream().synchronize()
t1 = time.time()
print(1.0 / (t1 - t0))
