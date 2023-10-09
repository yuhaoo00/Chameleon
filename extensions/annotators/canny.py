import cv2
import numpy as np
from PIL import Image

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        res = cv2.Canny(img, low_threshold, high_threshold)
        res = Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
        return res