import numpy as np
import cv2
from renderer import plot_sdf
from polygon_sampler import Polygon

SHAPE_PATH = '../shapes/shape/'
SHAPE_IMAGE_PATH = '../shapes/shape_images/'
HEATMAP_PATH = '../results/true_heatmaps/'

CANVAS_SIZE = np.array([800, 800])  # Keep two dimensions the same


class PolygonTrueRenderer(object):
    def __init__(self, polygon_name, polygon_path, polygon_image_path, res_path):
        self.polygon_name = polygon_name

        self.polygon_path = polygon_path
        self.polygon_image_path = polygon_image_path

        self.shape = Polygon()

        self.res_path = res_path

    def run(self):
        self.load()
        self.plot()

    def load(self):
        vertices = []
        f = open(f'{self.polygon_path}{self.polygon_name}.txt', 'r')
        line = f.readline()
        while line:
            x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
            vertices.append([x, y])
            line = f.readline()
        f.close()
        self.shape.set_v(np.array(vertices, dtype=np.double))

    def plot(self):
        # Plot_sdf
        plot_sdf(self.shape.sdf, 'cpu', res_path=HEATMAP_PATH, name=self.polygon_name, store_name = self.polygon_name,
                 is_net=False, show_image=False)


if __name__ == '__main__':
    print('Enter shape name:')
    shape_name = input()
    renderer = PolygonTrueRenderer(shape_name, SHAPE_PATH, SHAPE_IMAGE_PATH, HEATMAP_PATH)
    print('Plotting...')
    renderer.run()
    print('Done!')