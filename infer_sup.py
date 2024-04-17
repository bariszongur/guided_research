import os
import torch
from models.feature_training.model_semi_u import SemiSupU
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.mic_utils import test_iou
from torchvision import transforms
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--device', help='"cpu" if device is cpu, "cuda if the device is gpu"', default="cuda")
parser.add_argument('--syn_dir', help='Directory for synthetic image', default="/")
parser.add_argument('--syn_l1', help='Directory for first set of label', default="/")
parser.add_argument('--syn_l2', help='Directory for second set of label', default="/")
parser.add_argument('--model_path', help='Directory for the pretrained model', default="/")

p_args = parser.parse_args()
model_path = p_args.model_path
device = torch.device(p_args.device)

args_u = {
    "arch": "vit_small",
    "patch_size": 8,
    "num_classes": 0,
    "channels": [3, 32, 64, 128, 256, 256, 256, 256, 256],
    "scale_factor": 2,
    "dino_channels": 384,
    "W": 24,
    "H": 80

}

model = SemiSupU(args_u)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

dir_images = p_args.syn_dir
dir_label_r = p_args.syn_l1
dir_label_tr = p_args.syn_l2

image = cv2.imread(dir_images)
label_r = np.array(cv2.imread(dir_label_r))
label_tr = np.array(cv2.imread(dir_label_tr))
label_c = label_r + label_tr
label_c[label_c>0] = 1.
transform_norm = transforms.Compose([
transforms.ToTensor(),
transforms.Resize(size = (320, 320))
])
final_image = []
final_image_norm = []
full_ious = []
for i in range(5):
        temp_row = []
        temp_row_norm = []
        for j in range(6):
            image_0 = np.array(image)[i * 320:(i + 1) * 320, j * 320:(j + 1) * 320, :]
            image_0 = transform_norm(image_0)
            image_0 = image_0.unsqueeze(0)
            with torch.no_grad():
                feats = model(image_0.to(device)).squeeze()
                feats[feats<0.3] = 0.
                feats[feats>0.3] = 1.
                if image_0.mean()>0.90:
                    feats = torch.zeros(feats.shape)
                image_0 = torch.permute(image_0.squeeze(),(1, 2, 0))
            if j == 0:
                temp_row = feats.cpu().detach().numpy()
                temp_row_norm = image_0.cpu().detach().numpy()
            else:
                temp_row = np.append(temp_row, feats.cpu().detach().numpy(), axis=1)
                temp_row_norm = np.append(temp_row_norm, image_0.cpu().detach().numpy(), axis=1)
        if i == 0:
            final_image = temp_row
            final_image_norm = temp_row_norm
        else:
            final_image = np.append(final_image, temp_row, axis=0)
            final_image_norm = np.append(final_image_norm, temp_row_norm, axis=0)
full_ious.append(test_iou(torch.tensor(label_c[:,:,0]).unsqueeze(0).unsqueeze(0), torch.tensor(final_image).unsqueeze(0).unsqueeze(0)).numpy())

plt.imsave("extracted_mask.png", final_image)
print("IoU for the example image: ", np.mean(full_ious))
print("Extracted masks is saved.")
