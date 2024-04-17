from train_semi import main

semi_path = "./"
semi_label = "./"

args = {
    "u_args":{
        "arch" : "vit_small",
        "patch_size" : 8,
        "num_classes" : 0,
        "channels" : [3, 32, 64, 128, 256, 256, 256, 256, 256],
        "scale_factor" : 2,
        "dino_channels" : 384,
        "W" : 24,
        "H" :80
    
}
}

args_train = {
    "batch_size" : 8,
    "num_workers" : 0,
    "lr" : 1e-4,
    "lr_dino" : 1e-4,
    "max_epochs": 1000,
    "print_every_n": 1,
    "is_unet" : True,
    "experiment": "exp1",
    "experiment_name" : "exp1",
    "supervision_amount" : 150
}

main(semi_path, semi_label, args, args_train)



