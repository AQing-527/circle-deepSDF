import os
import numpy as np
import cv2
import torch

PTH_DATA_PATH = "../models/"
SAVE_PATH = "../models/model_images/"


def plot_weight(pth_data_path, save_path, shape_name, diff):
    if not diff:
        name = shape_name[0]
        data = torch.load(f'{pth_data_path}{name}.pth')
        i = 0
        for k, v in data.items():
            if i % 3 == 2:
                v = v.cpu().numpy()
                max_norm = np.max(np.abs(v))
                heat_map = v / max_norm * 127.5 + 127.5
                heat_map = np.uint8(heat_map)
                heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

                # Scale
                # heat_map = cv2.resize(heat_map, (map_size[0]*4, map_size[1]*4), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(f'{save_path}{name}_{i // 3 + 1}.png', heat_map)
                print(f'Heatmap path = {save_path}{name}_{i // 3 + 1}.png')

            i = i + 1
    else:
        name_1 = shape_name[0]
        name_2 = shape_name[1]
        data1 = torch.load(f'{pth_data_path}{name_1}.pth')
        data2 = torch.load(f'{pth_data_path}{name_2}.pth')
        items1 = list(data1.items())
        items2 = list(data2.items())
        for i in range(len(data1)):
            if i % 3 == 2:
                k1, v1 = items1[i]
                k2, v2 = items2[i]
                v1 = v1.cpu().numpy()
                # print(v1, flush=True)
                v2 = v2.cpu().numpy()
                # print(v2, flush=True)
                result = v1 - v2
                # print(result, flush=True)
                max_norm = np.max(np.abs(result))
                heat_map = result / max_norm * 127.5 + 127.5
                heat_map = np.uint8(heat_map)
                heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

                # Scale
                # heat_map = cv2.resize(heat_map, (map_size[0]*4, map_size[1]*4), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(f'{save_path}{name_1}_{name_2}_{i // 3 + 1}.png', heat_map)
                print(f'Heatmap path = {save_path}{name_1}_{name_2}_{i // 3 + 1}.png')


if __name__ == '__main__':
    difference = True
    name = []
    print('Enter shape name:')
    name.append(input())
    if difference:
        print('Enter another shape name:')
        name.append(input())

    print('Plotting results...')
    plot_weight(PTH_DATA_PATH, SAVE_PATH, name, difference)
    print('Done!')
