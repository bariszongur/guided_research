import matplotlib.pyplot as plt
from models.model_infer_experiments import InferModelExp
from models.mic_utils import *
from models.feature_training.model import  FeatureExtractor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='"cpu" if device is cpu, "cuda if the device is gpu"', default="cuda")
parser.add_argument('--is_ft', help='"F" if you want to work with original DINO model, "T" if you want to load a pretrained model', default="F")
parser.add_argument('--is_syn', help='"F" if you want to work with original images, "T" if Synthetic', default="F")
parser.add_argument('--model_path', help='Directory for the pretrained model', default="/")
parser.add_argument('--syn_dir', help='Directory for synthetic image if --is_syn is selected as "T"', default="/")
parser.add_argument('--mic_dir', help='Directory for microscopy image if --is_syn is selected as "F"', default="/")
parser.add_argument('--syn_l1', help='Directory for first set of label if --is_syn is selected as "T"', default="/")
parser.add_argument('--syn_l2', help='Directory for second set of label if --is_syn is selected as "T"', default="/")
parser.add_argument('--pca_dim', help='Number of dimensions to use in PCA, default should be 20 when working with microscopy images and 3 with synthetic images', default=3)
parser.add_argument('--color_weight', help='Concatanated color wight, default should be 0 when working with microscopy images and 0.3 - 1 with synthetic images', default=0.3)
p_args = parser.parse_args()
args = {
    "dino_args": {
        "arch": "vit_small",
        "patch_size": 8,
        "num_classes" : 0,
        "W" : 40,
        "H" : 40
    }
    ,
    "feature_extractor":{
        "patch_size" : 8,
        "channels" : 384,
        "scale_factor" : 2,
        "dino_channels" : 384,
        "K_transpose" : 2,
        "K_conv" : 3
    }
}


model_path = p_args.model_path
dir_images = p_args.syn_dir
dir_label_r = p_args.syn_l1
dir_label_tr = p_args.syn_l2
dir_images_real = p_args.mic_dir

if p_args.is_syn == "F":
    mic = True
else:
    if p_args.is_syn == "T":
        mic = False
    else:
        print("Wrong arguments")

if p_args.is_ft == "F":
    is_finetune = False
else:

    is_finetune = True


if is_finetune:
    model = FeatureExtractor(args)
    model.load_state_dict(torch.load(model_path))
    model = model.dino
else:
    model =vits.__dict__[args["dino_args"]["arch"]](patch_size=args["dino_args"]["patch_size"], num_classes=args["dino_args"]["num_classes"])

ious_color = []
ious_feat = []
if mic:
    image = cv2.imread(dir_images_real)
    image = resize_image_mic(image / 255., "bilinear")
    label_c = image
else:
    image = cv2.imread(dir_images)
    label_r = np.array(cv2.imread(dir_label_r))
    label_tr = np.array(cv2.imread(dir_label_tr))
    label_c = label_r + label_tr
    label_c[label_c > 0] = 1.

infer_model = InferModelExp(args, model, image, label_c, p_args.device)
iou_feat, labels_feats, gt_labels = infer_model.get_init_iou_scores("feats", 2, int(p_args.pca_dim), float(p_args.color_weight))
iou_color, labels_color, gt_labels = infer_model.get_init_iou_scores("color", 2, None, 0.1)
ious_color.append(iou_color)
ious_feat.append(iou_feat)

if not mic:
    ious_color_np = np.array(ious_color)
    ious_feat_np = np.array(ious_feat)
    print("IoU of Color-Based clustering: ", np.mean(ious_color_np))
    print("IoU of Feature-Based clustering: ", np.mean(ious_feat_np))

plt.imsave("mask_color.png", labels_color[0])
plt.imsave("mask_ssl.png", labels_feats[0])
plt.imsave("mask_color_inverted.png", labels_color[1])
plt.imsave("mask_ssl_inverted.png", labels_feats[1])
print("Extracted masks and inverse masks are saved.")