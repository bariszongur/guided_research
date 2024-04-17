import torch
#import models.vision_transformer as vits
import torch.nn as nn




class SemiSupU(nn.Module):


    """
    UNet Model implemented for semi-supervised training. Contains 5 encoder and 5 decoder blocks. Outputs the extracted mask.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=args["channels"][0], out_channels=args["channels"][1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(args["channels"][1]),
            nn.ReLU()
        )
        self.encoder1 = EncoderBlock(args["channels"][1], args["channels"][2])
        self.encoder2 = EncoderBlock(args["channels"][2], args["channels"][3])
        self.encoder3 = EncoderBlock(args["channels"][3], args["channels"][4])
        self.encoder4 = EncoderBlock(args["channels"][4], args["channels"][5])
        self.encoder5 = EncoderBlock(args["channels"][5], args["channels"][6])


        self.decoder1 = DecoderBlock(args["channels"][6], args["channels"][5], args["scale_factor"])
        self.decoder2 = DecoderBlock(args["channels"][5]*2, args["channels"][4], args["scale_factor"])
        self.decoder3 = DecoderBlock(args["channels"][4]*2, args["channels"][3], args["scale_factor"])
        self.decoder4 = DecoderBlock(args["channels"][3]*2 , args["channels"][2], args["scale_factor"])
        self.decoder5 = DecoderBlock(args["channels"][2]*2, args["channels"][1], args["scale_factor"])

        self.decoder6 = nn.Sequential(
            nn.Upsample(scale_factor=args["scale_factor"], mode="bilinear"),
            nn.Conv2d(in_channels=args["channels"][1], out_channels=1 , kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        enc1 = self.conv_block(x)
        enc2 = self.encoder1(enc1)
        enc3 = self.encoder2(enc2)
        enc4 = self.encoder3(enc3)
        enc5 = self.encoder4(enc4)

        enc6 = self.encoder5(enc5)

        dec5 = self.decoder1(enc6)
        dec4 = self.decoder2(torch.cat((dec5, enc5), 1))
        dec3 = self.decoder3(torch.cat((dec4, enc4), 1))
        dec2 = self.decoder4(torch.cat((dec3, enc3), 1))
        dec1 = self.decoder5(torch.cat((dec2, enc2), 1))

        output = self.decoder6(dec1)
        return output


class EncoderBlock(nn.Module):

    """
    Encoder block for the UNet.
    """
    def __init__(self,  in_ch, out_ch):
        super().__init__()
        self.in_ch= in_ch
        self.out_ch=out_ch

        self.enc_block=nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.enc_block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block for the UNet.
    """

    def __init__(self,  in_ch, out_ch, scale_factor):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.dec_block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
    def forward(self,x):
        return self.dec_block(x)
