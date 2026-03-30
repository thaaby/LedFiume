import cv2
import numpy as np

img = cv2.imread('/Users/nta11/Desktop/LedFiume/pixilart-drawing.png', cv2.IMREAD_UNCHANGED)
if img is not None:
    print("Shape:", img.shape)
    if len(img.shape) == 3:
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        print("Unique colors:", colors)
    else:
        print("Unique values:", np.unique(img))
else:
    print("Could not read image.")
