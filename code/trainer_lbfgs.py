import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net3_512 import SDFNet
from loader import SDFData
from renderer import plot_sdf

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
MODEL_PATH = '../models/'
RES_PATH = '../results/trained_heatmaps/'
SHAPE_IMAGE_PATH = '../shapes/shape_images/'
LOG_PATH = '../logs/'

if __name__ == '__main__':
    batch_size = 10000
    learning_rate = 1e-5
    epochs = 100
    curr_iterations = '0'
    show_frequency = 10
    regularization = 0  # Default: 1e-2
    delta = 0.1  # Truncated distance

    print('Enter shape name:')
    name = input()

    train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
    val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available(), flush=True)
    print(f'Using {device}!')

    model = SDFNet().to(device)
    if os.path.exists(f'{MODEL_PATH}{name}_{curr_iterations}.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}{name}_{curr_iterations}.pth'))

    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, gamma=0.1, last_epoch=-1)

    writer = SummaryWriter(LOG_PATH)
    total_train_step = 0
    total_val_step = 0

    start_time = time.time()
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')

        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        train_loss = 0
        for batch, (xy, sdf) in enumerate(train_dataloader):
            xy, sdf = xy.to(device), sdf.to(device)

            def closure():
                global xy, sdf, train_loss
                optimizer.zero_grad()
                predicted_sdf = model(xy)
                sdf = torch.reshape(sdf, predicted_sdf.shape)
                loss = loss_fn(torch.clamp(predicted_sdf, min=-delta, max=delta),
                               torch.clamp(sdf, min=-delta, max=delta))
                train_loss += loss.item()
                loss.backward()
                return loss


            optimizer.step(closure=closure)

            if batch % 1 == 0:
                # loss_num, current = loss.item(), batch * len(xy)
                print(f'{train_loss/size}')

            # total_train_step += 1
            # if total_train_step % 200 == 0:
            # writer.add_scalar('Training loss', loss.item(), total_train_step)
        total_train_step += 1
        train_loss /= size
        writer.add_scalar('Training loss', train_loss, total_train_step + int(curr_iterations))

        # Evaluation loop
        model.eval()
        size = len(val_dataloader.dataset)
        val_loss = 0

        with torch.no_grad():
            for xy, sdf in val_dataloader:
                xy, sdf = xy.to(device), sdf.to(device)
                predicted_sdf = model(xy)
                sdf = torch.reshape(sdf, predicted_sdf.shape)
                loss = loss_fn(torch.clamp(predicted_sdf, min=-delta, max=delta),
                               torch.clamp(sdf, min=-delta, max=delta))
                val_loss += loss.item()

        val_loss /= size
        end_time = time.time()
        print(f'Test Error: \n Avg loss: {val_loss:>8f} \n Time: {(end_time - start_time):>12f} \n ')

        total_val_step += 1
        writer.add_scalar('Val loss', val_loss, total_val_step + int(curr_iterations))

        # scheduler.step()

        if (t + 1) % show_frequency == 0:
            curr_name = name + '_' + str(t + 1 + int(curr_iterations))
            torch.save(model.state_dict(), f'{MODEL_PATH}{curr_name}.pth')
            plot_sdf(model, device, res_path=RES_PATH, name=name, store_name=curr_name, is_net=True,
                     show_image=False)

    print(f'Complete training with {epochs} epochs!')

    writer.close()

    # Plot results
    # print('Plotting results...')

    print('Done!')
