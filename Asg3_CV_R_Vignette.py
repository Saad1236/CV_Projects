import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the image from camera
img = cv2.imread('Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

### ADD Vignette ###

# Create a mask with the same size as the input image using np.zeros() to make a black image:
mask = np.zeros(img.shape[:2], dtype=np.float64)

# Create a white ellipse in the center of the mask
center_coordinates = (img.shape[1]//2, img.shape[0]//2)
axes_length = (img.shape[1]//2, img.shape[0]//2)
angle = 0
start_angle = 0
end_angle = 360
color = (255, 255, 255)
thickness = -2
cv2.ellipse(mask, center_coordinates, axes_length, angle, start_angle, end_angle, color, thickness)

mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


kernel_size = max(img.shape[0], img.shape[1])
sigma = -1
kernel = cv2.getGaussianKernel(kernel_size, sigma)
kernel = kernel * kernel.T

# Apply the kernel to the mask
mask = cv2.filter2D(mask, -1, kernel)

mask1 = cv2.merge([mask, mask, mask])

result = np.empty_like(img)
cv2.multiply(img, mask1, result, dtype=cv2.CV_8U)
result = cv2.convertScaleAbs(result, alpha=(255.0/np.max(result)), beta=1)

gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

rows, cols = gray.shape[:2]
x = np.linspace(-1, 1, cols)
y = np.linspace(-1, 1, rows)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)

vignette = 1 - (r**2 / np.max(r**2)) 
vignette = np.clip(vignette, 0, 1)
vignette = cv2.merge([vignette, vignette, vignette])


add = np.empty_like(img)
cv2.multiply(result, mask1, add, dtype=cv2.CV_8U)
#add1= cv2.cvtColor(add, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

axes[0].imshow(img)
axes[0].set_title('Normal Image')

axes[1].imshow(mask1)
axes[1].set_title('Mask')

axes[2].imshow(add)
axes[2].set_title('Vignette Image')

axes[3].imshow(result)
axes[3].set_title('Vignette Remove')
# Show the plot
plt.show()