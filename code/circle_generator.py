import numpy as np
import cv2
import time


CANVAS_SIZE = (800, 800)
SHAPE_COLOR = (255, 255, 255)
# Customize shape name for easier identification
SAVE_NAME = 'Circle_2'


class CircleDrawer(object):
    def __init__(self, window_name):
       self.window_name = window_name  # Name for the window
       self.x = -1
       self.y = -1
       self.radius = -1

    # (x,y) is the center of the circle
    def set_center(self, x, y):
        if (x > 0 and y > 0):
            self.center = (x, y)

    def set_radius(self, radius):
        if (radius > 0):
            self.radius = radius

    def run(self):
         cv2.namedWindow(self.window_name)
         cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
         canvas = np.zeros(CANVAS_SIZE, np.uint8)
         if (self.center[0] > 0 and self.center[1] > 0 and radius> 0):
            # generate the circle using the input parameters
             cv2.circle(canvas, self.center, self.radius, SHAPE_COLOR, thickness = 1)
             cv2.imshow(self.window_name, canvas)
         # Waiting for the user to press any key
         cv2.waitKey()     
         cv2.destroyWindow(self.window_name)
         return canvas

class DataSaver(object):
    def __init__(self, x, y, radius, image):
        self.x = x
        self.y = y
        self.radius = radius
        self.image = image
        self.data_path = '../shapes/shape/'
        self.image_path = '../shapes/shape_images/'
        self.save_name = f'circle_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_save_name(self, save_name):
        self.save_name = save_name

    def save(self):
        f = open(f'{self.data_path}{self.save_name}.txt', 'w')
        # Normalize (x,y) and radius to (0,1)
        x = self.x / CANVAS_SIZE[0]
        y = self.y / CANVAS_SIZE[1]
        # assume we always use square canvas
        radius = self.radius / CANVAS_SIZE[0]
        f.write(f'{x} {y} {radius}\n')

        f.close()

        cv2.imwrite(f'{self.image_path}{self.save_name}.png', self.image)

if __name__ == '__main__':
    drawer = CircleDrawer('press any key to exit and store this circle')
    x, y = input("Please enter the x, y coordinates of the center of this circle (the canvas size is 800*800): ").split()
    x = int(x)
    y = int(y)
    drawer.set_center(x, y)
    
    radius = int(input("Please enter the radius of this circle (the canvas size is 800*800): "))
    drawer.set_radius(radius)

    image = drawer.run()

    saver = DataSaver(x, y, radius, image)
    if SAVE_NAME != '':
        saver.set_save_name(SAVE_NAME)
    saver.save()
    print(f'Data path = {saver.data_path}{saver.save_name}.txt')
    print(f'Image path = {saver.image_path}{saver.save_name}.png')
    print("Done!")