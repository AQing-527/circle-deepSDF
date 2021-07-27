import numpy as np
import cv2
import logging

SHAPE_PATH = 'shapes/'
SHAPE_IMAGE_PATH = 'shapes/shape_images/'
SAMPLED_DATA_PATH = 'datasets/'
SAMPLED_IMAGE_PATH = 'datasets/sampled_images/'
CANVAS_SIZE = np.array([800, 800])  # Keep two dimensions the same
SHAPE_COLOR = (255, 255, 255)
POINT_COLOR = (127, 127, 127)


# The Shape and Circle classes are adapted from
# https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
class Shape:
    def sdf(self, p):
        pass
        
class Circle(Shape):
    
    def __init__(self, c, r):
        self.c = c
        self.r = r
    
    def set_c(self, c):
        self.c = c
    
    def set_r(self, r):
        self.r = r

    def sdf(self, p):
        return np.linalg.norm(p - self.c) - self.r

# The CircleSampler class is adapted from
# https://github.com/mintpancake/2d-sdf-net

class CircleSampler(object):
    def __init__(self, circle_name, circle_path, circle_image_path, sampled_data_path, sampled_image_path):
        self.circle_name = circle_name

        self.circle_path = circle_path
        self.circle_image_path = circle_image_path

        self.sampled_data_path = sampled_data_path
        self.sampled_image_path = sampled_image_path
        self.circle = Circle([0,0],0)

    def run(self, show_image):
        self.load()
        self.sample()
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


    def sample(self, m=4000, n=1000, var=(0.0025, 0.00025)):
        """
        :param m: number of points sampled on the boundary
                  each boundary point generates 2 samples
        :param n: number of points sampled uniformly in the unit circle
        :param var: two Gaussian variances used to transform boundary points
        """

        # Do uniform sampling
        x = np.random.uniform(0, 1, size=(n, 1))
        y = np.random.uniform(0, 1, size=(n, 1))
        uniform_points = np.concatenate((x, y), axis=1)

        # Do Gaussian sampling
        t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
        direction = np.concatenate((self.circle.r * np.cos(t), self.circle.r * np.sin(t)), axis=1)
        boundary_points = self.circle.c + direction

        # Perturbing boundary points
        noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=boundary_points.shape)
        noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=boundary_points.shape)
        gaussian_points = np.concatenate((boundary_points + noise_1, boundary_points + noise_2), axis=0)

        # Merge uniform and Gaussian points
        sampled_points = np.concatenate((uniform_points, gaussian_points), axis=0)
        self.sampled_data = self.calculate_sdf(sampled_points)

    def calculate_sdf(self, points):

        # Add a third column for storing sdf
        data = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
        data[:, 2] = np.apply_along_axis(self.circle.sdf, 1, data[:, :2])

        return data

    def save(self, show_image):

        save_name = f'sampled_{self.circle_name}'

        # Save sampled data to .txt
        f = open(f'{self.sampled_data_path}{save_name}.txt', 'w')
        for datum in self.sampled_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]}\n')
        f.close()

        if not show_image:
            return

        # Generate a sampled image
        window_name = 'Sampled Image'
        cv2.namedWindow(window_name)
        canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # Draw circle
        scaled_center = np.around(self.circle.c * CANVAS_SIZE).astype(int)
        # assume we always use square canvas
        scaled_radius = np.around(self.circle.r * CANVAS_SIZE[0]).astype(int)
        cv2.circle(canvas, scaled_center, scaled_radius, SHAPE_COLOR, thickness = 1)
        
        # Draw points
        for i, datum in enumerate(self.sampled_data):
            point = np.around(datum[:2] * CANVAS_SIZE).astype(int)
            cv2.circle(canvas, point, 1, POINT_COLOR, -1)
            if i % 50 == 0:
                radius = np.abs(np.around(datum[2] * CANVAS_SIZE[0]).astype(int))
                cv2.circle(canvas, point, radius, POINT_COLOR)

        # Store and show
        cv2.imwrite(f'{self.sampled_image_path}{save_name}.png', canvas)
        cv2.imshow(window_name, canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Enter shape name:')
    shape_name = input()
    sampler = CircleSampler(shape_name, SHAPE_PATH, SHAPE_IMAGE_PATH, SAMPLED_DATA_PATH, SAMPLED_IMAGE_PATH)
    print('Sampling...')
    sampler.run(show_image=True)
    print('Done!')