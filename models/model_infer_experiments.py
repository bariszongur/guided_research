from models.mic_utils import *
from sklearn.decomposition import PCA



'''
Main class for inference on unsupervised model. This class extracts the features and do both SSL feature based and 
color based clustering.
'''
class InferModelExp:

    def __init__(self, args, model, image, labels, device):
        '''
        args: Set of arguments as list for the model.
        model: DINO model instance
        image: Input image
        labels: Labels for the input image, values should be either 0 or 1
        '''

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        self.args = args
        self.device = torch.device(device)
        self.transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(320, 320)),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        self.image = image
        self.gt_labels_init = labels
        self.model = model

        self.attn = full_image_attn(self.model, self.image, self.device)
        self.feats = full_image_feats(self.model, self.image, self.device)
        self.resize_image = resize_image(self.image, "bilinear")
        self.gt_labels = resize_image(self.gt_labels_init, "nearest")

    def get_init_iou_scores(self, mode, k, pca, weight):
        '''
        Calculates IoU score and returns extracted mask for the image.

        mode: Either "attn" for SSL features of "color" for color based clustering
        pca: To apply pca or not. A bolean value
        weight: Weight of the color vectors concatanated to SSl features

        returns
        iou score for the image
        extracted mask
        ground truth labels
        '''
        gt_labels_ls = self.gt_labels[:,:,0]
        if mode == "feats":
            if pca is not None:
                pca = PCA(n_components=pca)
                pca_features = np.transpose(self.feats, (2, 0, 1))
                h, w = pca_features.shape[1], pca_features.shape[2]
                pca_features = np.reshape(pca_features,
                                          (pca_features.shape[0], pca_features.shape[1] * pca_features.shape[2]))
                pca.fit(pca_features)
                pca_features = pca.components_
                pca_features = np.reshape(pca_features, (pca_features.shape[0], h, w))
                pca_features = np.transpose(pca_features, (1, 2, 0))
                pca_features = np.concatenate((normalize(pca_features), normalize(self.resize_image)*weight), axis=2)
                labels_ls = k_means_labels(pca_features, k)
            else:
                features = np.concatenate((normalize(self.feats), normalize(self.resize_image) * weight), axis=2)
                labels_ls = k_means_labels(features, k)
        if mode == "attn":
            labels_ls = k_means_labels(self.attn, k)
        if mode == "color":
            labels_ls = k_means_labels(self.resize_image, k)
        ious = []


        for i in range(k):
            max_iou = 0
            it = np.logical_and(gt_labels_ls , labels_ls[i])
            un = np.logical_or(gt_labels_ls , labels_ls[i])
            iou = np.sum(it)/np.sum(un)
            if max_iou < iou:
                    max_iou = iou
                    ious.append(max_iou)
        return np.max(np.array(ious)), labels_ls , gt_labels_ls

    def calculate_iou_with_cluster_fg(self, cc1, cc2, th):

        '''
        Calculates IoU with given cluster centers
        cc1: First cluster center
        cc2: Second cluster center

        returns
        ious and extracted mask
        '''
        gt_labels_ls = self.gt_labels[:,:,0]
        feats_norm = torch.tensor(np.concatenate((normalize(self.feats), normalize(self.resize_image)), axis=2))
        #feats_norm = torch.nn.functional.normalize(feats_tensor, dim=2)
        l1 = torch.sum((feats_norm - cc1).pow(2), dim=2).sqrt().unsqueeze(0) >= th
        l2 = torch.sum((feats_norm - cc2).pow(2), dim=2).sqrt().unsqueeze(0) >= th
        labels_ls = torch.cat((l1, l2), dim=0).numpy()
        ious = []
        for i in range(2):
            max_iou = 0
            it = np.logical_and(gt_labels_ls, labels_ls[i])
            un = np.logical_or(gt_labels_ls, labels_ls[i])
            iou = np.sum(it) / np.sum(un)
            if max_iou < iou:
                max_iou = iou
            ious.append(max_iou)
        return np.array(ious), labels_ls

    def get_cluster_centers_fg(self):
        '''
        Calculates cluster centers for an alternative supervised approach

        returns

        cc1 and cc2 are 2 cluster centers
        '''
        feats_norm = torch.tensor(np.concatenate((normalize(self.feats), normalize(self.resize_image)), axis=2))
        gt_labels_ls = self.gt_labels[:, :, 0]
        gt_labels_ls = torch.tensor(np.array(gt_labels_ls))
        gt_labels_inv = (gt_labels_ls * -1) + 1
        ##feats_norm = torch.nn.functional.normalize(feats_tensor, dim=2)
        cc1 = (feats_norm * gt_labels_ls.unsqueeze(2))
        cc1 = torch.sum(cc1.reshape(cc1.shape[0] * cc1.shape[1], cc1.shape[2]), dim=0) / gt_labels_ls.sum()
        cc2 = (feats_norm * gt_labels_inv.unsqueeze(2))
        cc2 = torch.sum(cc2.reshape(cc2.shape[0] * cc2.shape[1], cc2.shape[2]), dim=0) / gt_labels_inv.sum()
        return cc1, cc2

