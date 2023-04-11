import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('abcd.jpg')
img = np.copy(image)
#img = cv2.flip(img, 1) #flip
#img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) #rotated
img = cv2.GaussianBlur(img, (5,5), 0)


# Convert color image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 50, 150, apertureSize=3)


lines = cv2.HoughLines(edges,       # Input edge image
                       1,           # Distance resolution in pixels
                       np.pi/120,   # Angle resolution in radians
                       190,         # Min number of votes for valid line
                       1)

for line in lines:
    rho, theta = line[0]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho
    x1, y1 = int(x0 + 10000*(-b)), int(y0 + 10000*(a))
    x2, y2 = int(x0 - 10000*(-b)), int(y0 - 10000*(a))
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

# Find the intersection points of the detected lines
points = []
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        rho1, theta1 = lines[i][0]
        rho2, theta2 = lines[j][0]
        
        # Check if the lines are parallel
        if np.abs(theta1 - theta2) < np.pi/180 :
            continue
        
        A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        points.append((x0, y0))

# Use a clustering algorithm to group the candidate vanishing points together
from sklearn.cluster import DBSCAN
X = np.array(points)
epsilon = 130
db = DBSCAN(eps=epsilon, min_samples=2).fit(X)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
centers = []
for i in range(n_clusters):
    cluster = X[labels == i]
    center = np.mean(cluster, axis=0)
    center = (int(center[0]), int(center[1]))
    centers.append(center)

# Draw circles around the vanishing points on the original image
for center in centers:
    cv2.circle(img, center, 10, (255, 0, 0), 5)


# Create a figure with four subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axes[0][0].imshow(image)
axes[0][0].set_title('Original Image')

axes[0][1].imshow(gray, cmap='gray')
axes[0][1].set_title('Grayscale')

axes[1][0].imshow(edges)
axes[1][0].set_title('Edges')

axes[1][1].imshow(img)
axes[1][1].set_title('Image with Vanishing Points')

# Show the plot
plt.show()
