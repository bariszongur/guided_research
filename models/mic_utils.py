import numpy as np
import cv2
import models.feature_training.vision_transformer as vits
from torchvision import transforms
import torch
from einops import rearrange
from sklearn.cluster import KMeans
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size = (320, 320)),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


def full_image_attn(model, image, device):

    """
    Returns the attention map for the input image

    model: ViT model
    image: input image
    device: cuda or cpu
    """
    model = model.to(device)
    final_image = []
    for i in range(5):
        temp_row = []
        for j in range(6):
            image_0 = np.array(image)[i * 320:(i + 1) * 320, j * 320:(j + 1) * 320, :]
            image_0 = transform_norm(image_0)
            image_0 = image_0.unsqueeze(0)
            attn = model.get_last_selfattention(image_0.to(device))
            nh = attn.shape[1]
            attn = attn[:, :, 0, 1:].reshape(nh, -1)
            attn = torch.permute(attn, (1, 0))
            attn = attn.reshape(40, 40, attn.shape[1])
            if j == 0:
                temp_row = attn.cpu().detach().numpy()
            else:
                temp_row = np.append(temp_row, attn.cpu().detach().numpy(), axis=1)
        if i == 0:
            final_image = temp_row
        else:

            final_image = np.append(final_image, temp_row, axis=0)
    return final_image

def full_image_feats(model, image, device):
    """
        Returns the feature map for the input image

        model: ViT model
        image: input image
        device: cuda or cpu
    """
    model = model.to(device)
    final_image = []

    for i in range(5):
        temp_row = []
        for j in range(6):
            image_0 = np.array(image)[i * 320:(i + 1) * 320, j * 320:(j + 1) * 320, :]
            image_0 = transform_norm(image_0)
            image_0 = image_0.unsqueeze(0)
            with torch.no_grad():
                feats = model.forward_all(image_0.to(device))
                feats = feats.squeeze().reshape(40, 40, feats.shape[2])
            if j == 0:
                temp_row = feats.cpu().detach().numpy()
            else:
                temp_row = np.append(temp_row, feats.cpu().detach().numpy(), axis=1)
        if i == 0:
            final_image = temp_row
        else:
            final_image = np.append(final_image, temp_row, axis=0)
    return final_image

def k_means_labels(image, k):
    """
       Apply k_means for the given set of features.
    """

    final_feats_kmeans = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    kmeans = KMeans(n_clusters=k).fit(final_feats_kmeans)
    labels = np.reshape(kmeans.labels_, (image.shape[0], image.shape[1]))
    labels_ls = []
    for i in range(k):
        labels_ls.append((labels == i) + 0)
    labels_ls = np.array(labels_ls)
    return labels_ls

def resize_image(image, mode):
    """
       Resizes the given image.
    """
    r_image = torch.permute(torch.tensor(image, dtype=torch.float32), (2, 0, 1)).unsqueeze(0)
    r_image = torch.nn.functional.interpolate(r_image, (image.shape[0] // 8, image.shape[1] // 8),
                                                  mode=mode)
    r_image = torch.permute(r_image.squeeze(), (1, 2, 0)).numpy()

    return r_image

def resize_image_mic(image, mode):
    """
    Resizes the given image.
    """


    r_image = torch.permute(torch.tensor(image, dtype=torch.float32), (2, 0, 1)).unsqueeze(0)
    r_image = torch.nn.functional.interpolate(r_image, (1600, 1920),
                                                  mode=mode)
    r_image = torch.permute(r_image.squeeze(), (1, 2, 0)).numpy()

    return r_image

def normalize(features):

    """
    Normalizes the given features.

    """
    features = (features - np.min(features)) / (
            np.max(features) - np.min(features))

    return features

def test_iou(gt_x, x):
    """
    Calculates the IoU for two given masks

    gt_x: Ground truth mask
    x: Generated Mask

    """

    x[x<0.5]=0.
    x[x>=0.5]=1.

    x_i = (x*-1) + 1

    #inter= gt_x + x 

    #inter[inter>=1.5]=1.
    #inter[inter<1.5]=0.

    #union= gt_x + x 
    #union[union>=0.5]=1.
    #union[union<0.5]=0.

    inter=torch.logical_and(gt_x, x)
    union=torch.logical_or(gt_x, x)
    
    inter = rearrange(inter, 'b c d h -> b (c d h)')
    union = rearrange(union, 'b c d h-> b (c d h)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)

    inter_i=torch.logical_and(gt_x, x_i)
    union_i=torch.logical_or(gt_x, x_i)
    
    inter_i = rearrange(inter_i, 'b c d h -> b (c d h)')
    union_i = rearrange(union_i, 'b c d h-> b (c d h)')

    iou_i = inter_i.sum(1) / (union_i.sum(1) + 1e-12)
    return torch.maximum(iou, iou_i)




