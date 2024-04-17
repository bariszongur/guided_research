from models.feature_training.dataloader_syn import SynDataset
import torch
from models.feature_training.model import FeatureExtractor
from pytorch_lightning.loggers import TensorBoardLogger

def train(model, train_dataloader, device, config):
    """
    Main training body for the fine-tune training of the DINO model.

    model: DINO model instance
    train_dataloader: dataloader for the training
    device: either cpu or cuda
    config: training arguments
    """


    logger = TensorBoardLogger("logs", name=config['experiment_name'])
    logger_index = 0
    logger.log_hyperparams(config)
    loss_criterion = torch.nn.L1Loss()
    loss_criterion.to(device)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1, verbose=False)
    train_loss_running = 0.
    n = 0
    model.train()
    best_loss = float('inf')
    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch.to(device)
            optimizer.zero_grad()
            recon_images = model(images)
            loss = loss_criterion(recon_images, images)
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
                    torch.save(model.state_dict(), fr'runs/{config["experiment"]}.ckpt')
                    best_loss = train_loss_running
        scheduler.step()


def main(dataset_path, args_model, args_train):

    """
    Main method for fine-tune training of the DINO model.

    dataset_path: Path of images for the dataset.
    args_model: Set of arguments for the model
    args_train: Training arguments
    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        print('Using CPU')

    dataset = SynDataset(dataset_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_train['batch_size'],
        shuffle=True,
        num_workers=args_train['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    model = FeatureExtractor(args_model)

    for name, param in model.dino.named_parameters():
        if not ("blocks.11" in name or "blocks.10" in name or "norm." in name):
            param.requires_grad = False

    model.to(device)
    train(model, dataloader, device, args_train)