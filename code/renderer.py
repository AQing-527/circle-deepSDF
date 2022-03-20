import os
import numpy as np
import cv2
import torch
from net4_512 import SDFNet

MODEL_PATH = '../models/'
SHAPE_IMAGE_PATH = '../shapes/shape_images/'
SHAPE_PATH = '../shapes/shape/'
RES_PATH = '../results/trained_heatmaps/'
POINTS_PATH = '../datasets/points/pred/'
EPOCHS = ''
WRITE_POINTS = False
IS_CIRCLE = False

# Adapted from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
def plot_sdf(sdf_func, device, res_path, name, store_name,
             img_size=800, is_net=False, show_image=False):
    # Sample the 2D domain as a regular grid
    low = 0
    high = 1
    grid_size = 800
    margin = 7e-3
    max_norm = 0.3  # Normalizing distance

    grid = np.linspace(low, high, grid_size + 1)[:-1]
    if not is_net:
        sdf_map = [[sdf_func(np.float_([x_, y_]))
                    for x_ in grid] for y_ in grid]
        sdf_map = np.array(sdf_map, dtype=np.float64)
    else:
        # Input shape is [1, 2]
        grid_points = [[[x, y] for y in grid for x in grid]]
        sdf_map = []
        sdf_func.eval()
        with torch.no_grad():
            for row in grid_points:
                row = torch.Tensor(row).to(device)
                row_sdf = sdf_func(row).detach().cpu().numpy()
                sdf_map.append(row_sdf)
        sdf_map = np.array(sdf_map)
        sdf_map = np.reshape(sdf_map, [grid_size, grid_size])

    # sdf_map = sdf_map[:-1, :-1]
    max_norm = np.max(np.abs(sdf_map)) if max_norm == 0 else max_norm
    heat_map = sdf_map / max_norm * 127.5 + 127.5
    heat_map = np.minimum(heat_map, 255)
    heat_map = np.maximum(heat_map, 0)

    # Plot predicted boundary
    low_pos = sdf_map > -margin
    high_pos = sdf_map < margin
    edge_pos = low_pos & high_pos
    if WRITE_POINTS:
        edge_indexes = np.where(edge_pos)
        if EPOCHS != '':
            f = open(f'{POINTS_PATH}{name}_{EPOCHS}.txt', 'w')
        else:
            f = open(f'{POINTS_PATH}{name}.txt', 'w')
        for edge_index in edge_indexes:
                f.write(f'{edge_index[0]} {edge_index[1]}\n')
        f.close()
    heat_map = np.where(edge_pos, 0, heat_map)

    # Scale to canvas size
    scale = int(img_size / grid_size)
    heat_map = np.kron(heat_map, np.ones((scale, scale)))

    # Generate a heat map
    # heat_map = None
    # heat_map = cv2.normalize(sdf_map, heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heat_map = np.uint8(heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    # Plot true boundary
    edge = cv2.imread(f'{SHAPE_IMAGE_PATH}{name}.png')
    heat_map = np.maximum(heat_map, edge)

    cv2.imwrite(f'{res_path}{store_name}.png', heat_map)
    print(f'Heatmap path = {res_path}{store_name}.png')

    if not show_image:
        return

    cv2.imshow('SDF Map', heat_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # print('Enter shape name:')
    # name = input()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')
    # if IS_CIRCLE:
    #     layers = 1
    # else:
    #     f = open(f'{SHAPE_PATH}{name}.txt', 'r')
    #     num = 0
    #     line = f.readline()
    #     while line:
    #         num = num + 1
    #         line = f.readline()
    #     f.close()
    #     if num // 7 < 3:
    #         layers = num // 7 + 2
    #     else:
    #         layers = 4
    model = SDFNet().to(device)
    # model = SDFNet(layers).to(device)
    names = ['Circle_10', 'Circle_100', 'Circle_1000', 'Circle_5000']
    for name in names:
        if EPOCHS != '':
            f_name = f'{MODEL_PATH}{name}_{int(EPOCHS)}.pth'
        else:
            f_name = f'{MODEL_PATH}{name}.pth'
        if os.path.exists(f_name):
            model.load_state_dict(torch.load(f_name))
        else:
            print('Error: No trained data!')
            exit(-1)

        print('Plotting results...')
        plot_sdf(model, device, res_path=RES_PATH, name=name, store_name=name+'_'+EPOCHS, is_net=True, show_image=False)
        print('Done!')
