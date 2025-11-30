import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    """Utility function to display an image."""
    plt.figure(figsize=(8,8))
    if len(image.shape) == 2: #Gray scale image
        plt.imshow(image,cmap='gray')
    else: # Color image
        plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

def interactive_edge_detection(image_path):
    """Interactive activity for edge detection and filtering."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error:Image not found")
        return
            
    # Convert to grayscale
    gray_image=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    display_image("Original Grayscale Image", gray_image)

    print("SELECT AN OPTION:")
    print("1.Sobel Edge detection")
    print("2.Canny Edge Detection")
    print("3.Laplacian Edge Detection")
    print("4.Gaussian Smoothing")
    print("5.Median Filtering")
    print("6.Exit")
    
    while True:
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            #Sobel Edge Detection
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F,1,0,ksize=3)
            sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=3)
            combined_sobel = cv2.bitwise_or(sobelx.astype(np.uint8), sobely.astype(np.uint8))
            display_image("Sobel Edge Detection", combined_sobel)

        elif choice == "2":
            #Canny Edge Detection
            print("Adjust thresholds for canny (default:100 and 200)")
            lower_thresh = int(input("Enter lower threshold: "))
            upper_thresh = int(input("Enter upper threshold: "))
            edges = cv2.Canny(gray_image, lower_thresh,upper_thresh)
            display_image("Canny Edge Detection",edges)
            
        elif choice == "3":
            #Laplacian Edge detection
            laplacian = cv2.laplacian(gray_image, cv2.CV_64F)
            display_image("Laplacian Edge Detection", np.abs(laplacian).astype(np.uint8))

        elif choice == "4":
            #Gaussian Smoothing
            print("Adjust kernal size for Gaussian blur (must be odd ,defsult:5)")
            kernal_size = int(input("Enter kernal size (odd number):"))
            blurred = cv2.GaussianBlur(image, (kernal_size,kernal_size),0)
            display_image("Gaussian Smoothed Image", blurred)

        elif choice == "5":
            # Median Filtering 
            print("Adjust kernal size for Median filtering (must be odd, default:5)")
            kernal_size = int(input("Enter kernal size(odd number): "))
            median_filered = cv2.medianBlur(image, kernal_size)
            display_image("Median Filtered Image", median_filered)

        elif choice == "6":
            print("Existing...")
            break
        
        else:
            print("Invalid choice. Please select a number between 1 and 6.")

# Provide the path to an image for the activity
interactive_edge_detection('BIRD.jpg')
