import cv2 as cv
import numpy as np

# what color do I want to use in my mask?
# -> look for one on the internet. Get the RGB code
limegreen = (50, 205, 50)
print(f'RGB - type = {type(limegreen)}\t\tvalue = {limegreen}')

# ok! I have the color. Now I need to convert it to HSV
limegreen_hsv = cv.cvtColor(src=cv.cvtColor(src=np.uint8([[limegreen]]), code=cv.COLOR_BGR2HSV), code=cv.COLOR_BGR2HSV)[0][0]
print(f'HSV - type = {type(limegreen_hsv)}\tvalue = {limegreen_hsv}')

# Create the interval of allowed color values

lower_bound = np.array([limegreen_hsv - 10, limegreen_hsv - 40, limegreen_hsv - 40])
upper_bound = np.array([limegreen_hsv + 10, limegreen_hsv + 40, limegreen_hsv + 40])

# And that's it! You can use it as boundary points