import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score
from trainer import train_model
import datahandler
import argparse
import os
import torch
import segmentation_models_pytorch as smp
import utilities as utils

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

# Set seed
torch.manual_seed(42)

model = smp.Unet(
    encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    classes=1,
    activation="sigmoid",          # model output channels (number of classes in your dataset)
)


# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
#criterion = torch.nn.MSELoss(reduction='mean')
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
criterion = smp.utils.losses.DiceLoss()

learning_rate = 0.001
encoder_learning_rate = 0.0001

# Since we use a pre-trained encoder, we will reduce the learning rate on it.
layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

# This function removes weight_decay for biases and applies our layerwise_params
model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

# Specify the optimizer and LR scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.1, patience=2,verbose=True)

# Specify the optimizer with a lower learning rate
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score, 'IoU': jaccard_score}

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# Create the dataloader
dataloaders = datahandler.get_dataloader_single_folder(
    data_dir, batch_size=batchsize)


# Run Trainer    
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, scheduler, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'state_dict.pt'))
torch.save(model, os.path.join(bpath, 'weights.pt'))