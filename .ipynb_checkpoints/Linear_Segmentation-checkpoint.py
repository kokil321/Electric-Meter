import cv2
import numpy as np
import base64
from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2lab, rgb2hsv
from PIL import Image


def apply_slic(img_b64):
    
    image_binary = base64.b64decode(img_b64)
    # Convert the binary data to an OpenCV image
    nparr = np.frombuffer(image_binary, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segments = slic(rgb_image, n_segments=2, compactness=10)
    
    # Find the label of the middle segment (assuming there are only two segments)
    unique_labels, label_counts = np.unique(segments, return_counts=True)
    middle_segment_label = unique_labels[np.argmax(label_counts)]
    
    try:
        # Find the coordinates of the bounding box for the middle segment
        rows, cols = np.where(segments == 1)
        s1x1, s1y1 = min(cols), min(rows)
        s1x2, s1y2 = max(cols), max(rows)

        # Find the coordinates of the bounding box for the middle segment
        rows, cols = np.where(segments == 2)
        s2x1, s2y1 = min(cols), min(rows)
        s2x2, s2y2 = max(cols), max(rows)
    except Exception as e:
        cropped_image = image
    else:
        # Crop the image based on the bounding box
        if s1x1 > s2x1 :
            cropped_image = image[s1y1:s1y2, s1x1:s1x2]
        else :
            cropped_image = image[s2y1:s2y2, s2x1:s2x2]
    
    _, encoded_image = cv2.imencode('.jpeg', cropped_image)
    b64_img = base64.b64encode(encoded_image).decode('utf-8')
 
    return b64_img



def stringToImage(base64_string):
    image_binary = base64.b64decode(base64_string)
    # Convert the binary data to an OpenCV image
    nparr = np.frombuffer(image_binary, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
    return image

def ImageTostring(image):
    # Encode the image as base64
    _, encoded_image = cv2.imencode('.jpg', image)
    b64_img = base64.b64encode(encoded_image).decode('utf-8')
    return b64_img