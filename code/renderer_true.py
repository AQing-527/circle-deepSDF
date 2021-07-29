import numpy as np
import cv2
from renderer import plot_sdf
from sampler import Circle

SHAPE_PATH = '../shapes/shape/'
SHAPE_IMAGE_PATH = '../shapes/shape_images/'
HEATMAP_PATH = '../results/true_heatmaps/'

CANVAS_SIZE = np.array([800, 800])  # Keep two dimensions the same

# The CircleSampler class is adapted from
# https://github.com/mintpancake/2d-sdf-net
class TrueRenderer(object):
    def __init__(self, circle_name, circle_path, circle_image_path, res_path, show_image=False ):
        self.circle_name = circle_name

        self.circle_path = circle_path
        self.circle_image_path = circle_image_path

        self.circle = Circle([0,0],0)
        
        self.show_image = show_image

    def run(self, show_image):
        self.load()
        self.save(show_image)

    # load the coordinate of center and the radius of the circle for sampling
    def load(self):
        f = open(f'{self.circle_path}{self.circle_name}.txt', 'r')
        line = f.readline()
        x, y, radius = map(lambda n: np.double(n), line.strip('\n').split(' '))
        center = np.array([x, y])

        f.close()
        self.circle.set_c(center)
        self.circle.set_r(radius)

    def save(self, show_image):
        # Plot_sdf
        plot_sdf(self.circle.sdf, 'cpu', res_path=HEATMAP_PATH, name=self.circle_name, shape_path=SHAPE_IMAGE_PATH, is_net=False, show=False)

        if not show_image:
            return

if __name__ == '__main__':
    print('Enter shape name:')
    shape_name = input()
    renderer = TrueRenderer(shape_name, SHAPE_PATH, SHAPE_IMAGE_PATH, HEATMAP_PATH)
    print('Plotting...')
    renderer.run(False)
    print('Done!')