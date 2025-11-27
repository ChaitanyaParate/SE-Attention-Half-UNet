import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import Att_H_UNET
from loss import BCEDiceLoss
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 12
NUM_EPOCHS = 120
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256 
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "Kvasir-SEG/images"
TRAIN_MASK_DIR = "Kvasir-SEG/masks"
VAL_IMG_DIR = "sessile-main-Kvasir-SEG/images"
VAL_MASK_DIR = "sessile-main-Kvasir-SEG/masks"

def train_fn(loader, model, optimizer, loss_fn, scaler, scheduler,epoch):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            predictions = model(data)
            

            loss = loss_fn(predictions, targets)


        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())
    scheduler.step()

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = Att_H_UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    dice_history = []
    best_dice = 0.9
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               T_0=5,        # restart every 5 epochs
                                                               T_mult=2,     # restart frequency doubles
                                                               eta_min=1e-7  # minimum LR
                                                               )
    
    load_checkpoint("my_checkpoint.pth.tar", model, optimizer)
    dice_history = torch.load("dice_history.pt")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler, epoch)
        

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        dice = check_accuracy(val_loader, model, device=DEVICE)
        if dice > best_dice:
            save_checkpoint(checkpoint)
            best_dice = dice

        dice_history.append(dice.item())

        torch.save(dice_history, "dice_history.pt")

if __name__ == "__main__":
    main()