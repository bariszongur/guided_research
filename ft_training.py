from train_syn import main


path = "./blenderconfig/renders/split/"

args = {
    "dino_args": {
        "arch": "vit_small",
        "patch_size": 8,
        "num_classes": 0,
        "W": 40,
        "H": 40
    }
    ,
    "feature_extractor": {
        "patch_size": 8,
        "channels": 384,
        "scale_factor": 2,
        "dino_channels": 384,
        "K_transpose": 2,
        "K_conv": 3
    }
}
args_train = {
    "batch_size": 8,
    "num_workers": 0,
    "lr": 1e-4,
    "lr_dino": 1e-4,
    "max_epochs": 1000,
    "print_every_n": 1,
    "is_unet": True,
    "experiment": "exp1",
    "experiment_name": "exp1",
    "supervision_amount": 150
}

main(path, args, args_train)



