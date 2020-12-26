import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score
from trainer import train_model
import datahandler
import argparse
import os
import torch
import segmentation_models_pytorch as smp

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--batchsize", default=16, type=int)

args = parser.parse_args()


bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize

model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    classes=1,
    activation="sigmoid",          # model output channels (number of classes in your dataset)
)
model.train()

#preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet18","imagenet")

# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
#criterion = torch.nn.MSELoss(reduction='mean')
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
criterion = smp.utils.losses.DiceLoss()

# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score, 'IoU': jaccard_score}


# Create the dataloader
dataloaders = datahandler.get_dataloader_single_folder(
    data_dir, batch_size=batchsize)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'state_dict.pt'))
torch.save(model, os.path.join(bpath, 'weights.pt'))

# Export ONNX format 
onnxmodel = trained_model.cuda().eval().half()
x = torch.ones(1, 3, 320, 640, requires_grad=True).cuda().half()
torch_out = model(x)
torch.onnx.export(onnxmodel,x,os.path.join(bpath, 'jetracer.onnx'),export_params=True,opset_version=11,do_constant_folding=True)
