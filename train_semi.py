from models.feature_training.dataloader_semisup import SynDatasetUns
import torch
from models.feature_training.model_semi import SemiSegmenter
from pytorch_lightning.loggers import TensorBoardLogger
from models.feature_training.model_semi_u import SemiSupU
import random

def train(model, train_dataloader, device, config):


    """
    Training loop of the UNet semi-supervised training model.


    train_dataloader: Dataloader for the training
    device: either cuda or cpu
    config: training arguments
    """
    logger = TensorBoardLogger("logs", name=config['experiment_name'])
    logger_index = 0
    logger.log_hyperparams(config)
    loss_criterion = torch.nn.BCELoss()
    loss_criterion.to(device)
    if config["is_unet"]:
         params = [
            {
                'params': model.parameters(),
                'lr': config['lr']
            }]
    else:
        params = [
            {
                'params': model.dino.parameters(),
                'lr': config['lr_dino']
            },

            {
                'params': model.upsampler.parameters(),
                'lr': config['lr']
            },
            {
                'params': model.last_layer.parameters(),
                'lr': config['lr']
            }

        ]
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5, last_epoch=-1, verbose=False)
    train_loss_running = 0.
    n = 0
    model.train()
    best_loss = float('inf')
    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch["image"].to(device)
            
            labels = torch.tensor(batch["label"][:,:,:,0], dtype=torch.float32).to(device)
            #labels = torch.reshape(labels, (labels.shape[0], labels.shape[1] * labels.shape[2]))
            optimizer.zero_grad()
            recon_images = model(images).squeeze()
            #recon_images = torch.reshape(recon_images, (recon_images.shape[0], recon_images.shape[1] * recon_images.shape[2]))
            loss = loss_criterion(recon_images, labels)
            loss.backward()
            optimizer.step()
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx
            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                logger_index = logger_index + 1
                n = n + 1
                samples_per_epoch = config["print_every_n"]
                train_loss_running = train_loss_running / samples_per_epoch
                print(f"[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / samples_per_epoch:.6f}")
                logger.log_metrics({
                    'train_loss': train_loss_running,
                    'epoch': epoch
                }, logger_index)
                train_loss_running = 0.
                if train_loss_running < best_loss:
                    best_loss = train_loss_running
        torch.save(model.state_dict(), fr'runs/{config["experiment"]}.ckpt')
        scheduler.step()


def main(dataset_path, label_path, args_model, args_train):

    """
    Main function of the semi-supervised training.

    dataset_path: Path for the input images.
    label_path: Path for ground truth labels.
    args_model: Arguments for the UNet Model.
    args_train: Arguments for training.

    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        print('Using CPU')

    dataset = SynDatasetUns(dataset_path, label_path)
    indices = random.sample(range(0, len(dataset)), args_train['supervision_amount'])
    data_sub = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        data_sub,
        batch_size=args_train['batch_size'],
        shuffle=True,
        num_workers=args_train['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    if args_train["is_unet"]:
        model = SemiSupU(args_model["u_args"])
    else:
        model = SemiSegmenter(args_model)
        for name, param in model.dino.named_parameters():
            if not ("blocks.11" in name or "blocks.10" in name or "norm." in name):
                param.requires_grad = False

    
    

    model.to(device)
    train(model, dataloader, device, args_train)
if __name__ == '__main__':
    main()
