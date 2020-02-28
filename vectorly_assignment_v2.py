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
    median_mask = cv2.inRange(1*median_blur, lower, upper) # TOGGLE the upper_range and lower_range 
    
    # Dilating the image to compensate for the loss of some (boundary)pixels due to blurring.
    kernel = np.ones((5,5), np.uint8)
    Median_Mask_dilation = cv2.dilate(median_mask, kernel, iterations=1)
    
    # Additional ways to improve the output
    Median_Mask_dilation_opening = cv2.morphologyEx(Median_Mask_dilation, cv2.MORPH_OPEN, kernel) 
    Median_Mask_dilation_closing  = cv2.morphologyEx(Median_Mask_dilation, cv2.MORPH_CLOSE, kernel) 
    
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
    return output

if __name__ == '__main__':

    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    cv2.imwrite('simpons_text.png',output)
    
