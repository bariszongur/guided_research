import torch.nn as nn
import models.feature_training.vision_transformer as vits



class Block(nn.Module):
    """
    Upsampler block used in SemiSegmenter model.
    """
    def __init__(self, in_ch, out_ch, K_transpose, K_conv):
        super().__init__()
        self.block = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=K_transpose, stride=2),
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=K_conv, stride=1, padding=1),
        nn.BatchNorm2d(num_features=out_ch),
        nn.LeakyReLU()
        )
    def forward(self,x):
        return self.block(x)
class SemiSegmenter(nn.Module):
    """

    Segmenter model that is built for semi-supervised cases. Uses 3 upsampler heads on top of DINO features.

    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.f_args = args["feature_extractor"]
        self.dino = vits.__dict__[args["dino_args"]["arch"]](patch_size=args["dino_args"]["patch_size"], num_classes=args["dino_args"]["num_classes"])
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.f_args["channels"]//4, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.upsampler = nn.Sequential(
            Block(self.f_args["channels"], self.f_args["channels"]//2, self.f_args["K_transpose"], self.f_args["K_conv"]),
            Block(self.f_args["channels"]//2,  self.f_args["channels"]//4, self.f_args["K_transpose"], self.f_args["K_conv"]),
            Block(self.f_args["channels"]//4, self.f_args["channels"]//4, self.f_args["K_transpose"], self.f_args["K_conv"]),
        )

    def forward(self, x):
        dino_feats = self.dino.forward_all(x)
        dino_feats = dino_feats.transpose(1, 2)
        dino_feats = dino_feats.reshape(x.shape[0], dino_feats.shape[1], self.args["dino_args"]["W"],
                                        self.args["dino_args"]["H"])
        return self.last_layer(self.upsampler(dino_feats))

    def forward_features(self, x):
        dino_feats = self.dino.forward_all(x)
        dino_feats = dino_feats.transpose(1, 2)
        dino_feats = dino_feats.reshape(x.shape[0], dino_feats.shape[1], self.args["dino_args"]["W"],
                                        self.args["dino_args"]["H"])
        return self.upsampler(dino_feats)
    def forward_dino_features(self, x):
        dino_feats = self.dino.forward_all(x)
        dino_feats = dino_feats.transpose(1, 2)
        dino_feats = dino_feats.reshape(x.shape[0], dino_feats.shape[1], self.args["dino_args"]["W"],
                                        self.args["dino_args"]["H"])
        return dino_feats



