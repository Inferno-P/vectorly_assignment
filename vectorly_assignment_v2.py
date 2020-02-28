import cv2
import numpy as np
import imutils

def getTextOverlay(input_image):

    # Scaling Image
    R,C = input_image.shape[1], input_image.shape[0]
    scaling_factor = 1.0
    new_R, new_C = int(R*scaling_factor), int(C*scaling_factor)
    resized_img = cv2.resize(input_image, (new_R, new_C))
    
    print("Old Size = ", input_image.shape)
    print("New Size = ", resized_img.shape)
    
    # Trying Median filtering 
    median_blur =  cv2.medianBlur(resized_img,7) # Changing the kernel size can have significant effects 

    # Filtering basis a RGB range
    lower = np.array([0, 0, 0])
    upper = np.array([20, 20, 20])
    median_mask = cv2.inRange(2*median_blur, lower, upper) # TOGGLE the upper_range and lower_range 
    
    # Dilating the image to compensate for the loss of some (boundary)pixels due to blurring.
    kernel = np.ones((5,5), np.uint8)
    Median_Mask_dilation = cv2.dilate(median_mask, kernel, iterations=1)  # Version 0 solution
    
    # Additional ways to improve the output 
    # Dilation and Opening are morphological operations that tend to be helpful in 'polishing' various irrelevant # artifacts like bumps on the outer boundary and depressions on the inner boundary.
    Median_Mask_dilation_opening = cv2.morphologyEx(Median_Mask_dilation, cv2.MORPH_OPEN, kernel)  # Improvement
    Median_Mask_dilation_closing  = cv2.morphologyEx(Median_Mask_dilation, cv2.MORPH_CLOSE, kernel) # Improvement 
    
    """
    while True :
        
        # Projecting the Median Mask
        #cv2.imshow("Median ", median_blur)
        #cv2.imshow("Median Masking", 255 - Median_Mask_dilation)
        #cv2.imshow("Median_Mask_dilation_opening", 255-Median_Mask_dilation)
        #cv2.imshow("Median_Mask_dilation_closing", 255 -  Median_Mask_dilation_closing)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    """ 
    
    
    output = 255 - Median_Mask_dilation_closing
    
    
    
    """
    Observation : To the best of my knowledge, vectorly aims to achieve most satisfying results with minimum processing. Hence, I have tried to provide my solution using the quite basic techniques in Image Processing
    which are not compute-intensive. 
    
    Next steps would be using algorithms that draw inferences based on the structures within the pixel groups rather than colorspaces. 
    For eg: 
    Alternate Approach:
        1. Converting the image into a grayscale.
        
        2. Applying Canny Edge.
            2.1 Applying Dilation and Opening operations to resolve broken boundary issues, if required.
        
        3. Performing Contour tracing. Since edges of pixel groups inside text characters form closed polygons, we can count the number of contours that form closed polygons.
        
        4. Transfering all the closed contours into a new image with white background and filling them with black colors.
    
    """
    return output

if __name__ == '__main__':

    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    cv2.imwrite('simpons_text.png',output)
    
