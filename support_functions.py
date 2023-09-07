""" IMPORT LIBIRARIES """

import cv2
import numpy as np
import matplotlib.pyplot as plt


import pytesseract
import re

from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example path on Windows



""" DEFINE FUNCTIONS """
def reshape_img(image,height, width):
    img_tmp = image.copy()
    img_tmp = cv2.resize(img_tmp,(width,height))
    return img_tmp


def crop_rect(image, x, y, crop_height, crop_width, outfile):
    img_tmp = image.copy()
    img_tmp = img_tmp[y:y + crop_height, x:x + crop_width]
    
    plt.imshow(img_crp)
    cv2.imwrite('./Crop_Images/'+ outfile, img_tmp)
    return img_tmp


def apply_template_matching(image, template):
    full = image.copy()
    face = template.copy()
    
    full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED' ]
    add_methods = ['cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    height, width,channels = template.shape
    
    for m in methods:

        # Create a copy of the image
        full_copy = full.copy()

        # Get the actual function instead of the string
        method = eval(m)

        # Apply template Matching with the method
        res = cv2.matchTemplate(full_copy,face,method)

        # Grab the Max and Min values, plus their locations
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Set up drawing of Rectangle

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        # Notice the coloring on the last 2 left hand side images.
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc    
        else:
            top_left = max_loc

        # Assign the Bottom Right of the rectangle
        bottom_right = (top_left[0] + width, top_left[1] + height)
        print(bottom_right) 

        # Draw the Red Rectangle
        cv2.rectangle(full_copy,top_left, bottom_right, 255, 10)
        
        # Crop the image 
        img_tmp = full.copy()
        img_tmp = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Plot the Images
        plt.subplot(121)
        plt.imshow(full_copy)
        plt.title('Detected Point')

        plt.subplot(122)
        plt.imshow(img_tmp)
        plt.title('cropped image')
        plt.suptitle(m)


        plt.show()
        print('\n')
        print('\n')

  
def apply_edge_detection(image, lower_threshold, upper_threshold):
    
    img_tmp = image.copy()
    #Convert the image to RGB from BGR 
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Convert the image to BGR to Gray 
    gray_image = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
      
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
    
    return edges

