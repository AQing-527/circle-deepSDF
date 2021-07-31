import numpy as np
import cv2
import time


CANVAS_SIZE = (800, 800)
SHAPE_COLOR = (255, 255, 255)
# Customize shape name for easier identification
SAVE_NAME = 'Noise_Circle_3'
AMPLITUDE = 20
POINTS_NUM = 20

class CircleDrawer(object):
    def __init__(self, window_name):
       self.window_name = window_name  # Name for the window
       self.x = -1
       self.y = -1
       self.radius = -1
       self.points = []

    # (x,y) is the center of the circle
    def set_center(self, x, y):
        if (x > 0 and y > 0):
            self.center = (x, y)

    def set_radius(self, radius):
        if (radius > 0):
            self.radius = radius

    def run(self, noise):
         cv2.namedWindow(self.window_name)
         cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
         canvas = np.zeros(CANVAS_SIZE, np.uint8)
         if (self.center[0] > 0 and self.center[1] > 0 and radius> 0):
             if (noise == False):
                # generate the circle using the input parameters
                cv2.circle(canvas, self.center, self.radius, SHAPE_COLOR, thickness = 1)
                cv2.imshow(self.window_name, canvas)
            
             else:
                for i in range(POINTS_NUM):
                    self.points.append([int(self.center[0]+(self.radius+AMPLITUDE)*np.cos((2*np.pi/POINTS_NUM)*i)), int(self.center[1]+(self.radius+AMPLITUDE)*np.sin((2*np.pi/POINTS_NUM)*i))])
                    self.points.append([int(self.center[0]+(self.radius-AMPLITUDE)*np.cos((2*np.pi/POINTS_NUM)*i+(1/POINTS_NUM)*np.pi)), int(self.center[1]+(self.radius-AMPLITUDE)*np.sin((2*np.pi/POINTS_NUM)*i+(1/POINTS_NUM)*np.pi))])
                cv2.polylines(canvas, [np.array(self.points)], isClosed=True, color = SHAPE_COLOR, thickness=1)
                cv2.imshow(self.window_name, canvas)
            
         # Waiting for the user to press any key
         cv2.waitKey()     
         cv2.destroyWindow(self.window_name)
         return self.points, canvas

class DataSaver(object):
    def __init__(self, x, y, radius, points, image):
        self.x = x
        self.y = y
        self.radius = radius
        self.points = points
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

    def save(self, noise):
        f = open(f'{self.data_path}{self.save_name}.txt', 'w')
        if not noise:
            # Normalize (x,y) and radius to (0,1)
            x = self.x / CANVAS_SIZE[0]
            y = self.y / CANVAS_SIZE[1]
            # assume we always use square canvas
            radius = self.radius / CANVAS_SIZE[0]
            f.write(f'{x} {y} {radius}\n')

            f.close()

            cv2.imwrite(f'{self.image_path}{self.save_name}.png', self.image)

        else:
            for point in self.points:
                # Normalize (x,y) to (0,1)
                x = np.double(np.double(point[0]) / np.double(CANVAS_SIZE[0]))
                y = np.double(np.double(point[1]) / np.double(CANVAS_SIZE[1]))
                f.write(f'{x} {y}\n')
                
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

    points, image = drawer.run(noise=True)

    saver = DataSaver(x, y, radius, points, image)
    if SAVE_NAME != '':
        saver.set_save_name(SAVE_NAME)
    saver.save(noise=True)
    print(f'Data path = {saver.data_path}{saver.save_name}.txt')
    print(f'Image path = {saver.image_path}{saver.save_name}.png')
    print("Done!")