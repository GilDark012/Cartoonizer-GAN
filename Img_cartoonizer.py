# Inspiration: https://www.youtube.com/watch?v=2xqvGZS7NCw
#%% IMPORT lIBRARY  
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# %% Load and rename

def image_read(path):
    """
    Reads an image from the given path and converts it to RGB.
    
    Args:
        path (str): Path to the image file
    
    Returns:
        np.ndarray: The image in RGB format, or None if the image could not be loaded
    """
    image = cv2.imread(path)
    if image is not None:  # Check if the image is successfully loaded
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        print(f"Failed to load image at {path}")
        return None

def rename_and_load_images(folder_path):
    """
    Renames images in a directory and loads them into a list.
    
    Args:
        folder_path (str): Path to the directory containing the images
    
    Returns:
        list of np.ndarray: List of images in RGB format
    """
    images = []
    image_files = [
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))
    ]
    image_files.sort()  # Ensure consistent ordering
    
    for idx, filename in enumerate(image_files, start=1):
        # Construct the new filename
        new_filename = f"img_{idx:03d}.jpg"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # Check if new path already exists before renaming
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")
        else:
            print(f"File {new_filename} already exists, skipping rename")
            new_path = old_path  # Use the old path if renaming is skipped

        # Load the renamed image
        image = image_read(new_path)
        if image is not None:
            images.append(image)

    return images

def display_images(images, cartoons=[], num_images=10):
    """
    Displays the first num_images in a grid using subplot.
    
    Args:
        images (list of np.ndarray): List of images in RGB format
        cartoons (list of np.ndarray, optional): List of cartoon images in RGB format. Defaults to [].
        num_images (int, optional): Number of images to display. Defaults to 10.
    """
    num_images = min(num_images, len(images))  # Limit to available images
    cols = 2  # Two columns: original and grayscale
    rows = num_images  # One row per image (original and grayscale)

    plt.figure(figsize=(10, rows * 5))  # Adjust figure size for better visibility

    for i in range(num_images):
        # Original image
        plt.subplot(rows, cols, 2 * i + 1)
        plt.imshow(images[i])  # Convert BGR to RGB for display
        plt.axis('off')
        plt.title(f"Original {i + 1}")
        
        # Cartoon image
        if cartoons and len(cartoons) > 0:            
            cartoon = cartoons[i]
            plt.subplot(rows, cols, 2 * i + 2)
            plt.imshow(cartoon)
            plt.axis('off')
            plt.title(f"Cartoon {i + 1}")

    plt.tight_layout()
    plt.show()

#%% Edge mask

def edge_mask(images, line_size, blur_value):
    """
    Applies edge detection to a list of images using adaptive thresholding.
    
    Args:
        images (list of np.ndarray): List of images in BGR format
        line_size (int): Block size for adaptive thresholding. Must be odd.
        blur_value (int): Kernel size for median blurring. Must be odd.
    
    Returns:
        list of np.ndarray: List of edge-detected images in grayscale
    """
    edges_img = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value
        )
        edges_img.append(edges)
    return edges_img

#%% Color Palette

def color_quantization(images, k):
    """
    Performs color quantization on an image using k-means clustering.
    
    Args:
        images (list of np.ndarray): List of images in RGB format
        k (int): Number of clusters for k-means clustering
    
    Returns:
        list of np.ndarray: List of color-quantized images in RGB format
    """
    results_img = []
    for img in images:
        # Transform the image
        data = np.float32(img).reshape((-1, 3))
        #Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        #Implemting k-means
        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        results_img.append(result)
        # plt.imshow(result)
        # plt.show()
    return results_img

#%% Reduice Noise

def noise_removal(result):
    """
    Reduces noise in an image using bilateral filtering.
    
    Args:
        result (list of np.ndarray): List of images in RGB format
    
    Returns:
        list of np.ndarray: List of noise-reduced images in RGB format
    """
    blurred_img=[]
    for img in result:
        # result = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        blurred = cv2.bilateralFilter(img, d=3, sigmaColor=200,sigmaSpace=200)
        blurred_img.append(blurred)
        
    return blurred_img

#%% Cartoonize

def cartoonize(blurred, edge):
    """
    Cartoonizes an image by combining edge detection and color quantization.
    
    Args:
        blurred (list of np.ndarray): List of blurred images in RGB format
        edge (list of np.ndarray): List of edge-detected images in grayscale
    
    Returns:
        list of np.ndarray: List of cartoonized images in RGB format
    """
    Cartoons_img = []
    for i, img in enumerate(blurred):
        cartoon = cv2.bitwise_and(img, img, mask=edge[i])
        Cartoons_img.append(cartoon)

    return Cartoons_img

# %% SAVING cartoon image

def save_cartoons(cartoons, output_folder):
    """
    Saves cartoon images to the specified output folder.
    
    Args:
        cartoons (list of np.ndarray): List of cartoon images to save
        output_folder (str): Path to folder where images will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Save each cartoon image
    for i, cartoon in enumerate(cartoons):
        # Convert RGB to BGR for saving
        cartoon_bgr = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        
        # Construct output filename
        output_path = os.path.join(output_folder, f"cartoon_{i+1:03d}.jpg")
        
        # Save the image
        cv2.imwrite(output_path, cartoon_bgr)
        print(f"Saved cartoon image to {output_path}")

# %% Execution 

def exec(save=False):
    """
    Executes the cartoonization process.
    
    Args:
        save (bool, optional): Whether to save the cartoonized images. Defaults to False.
    """
    folder_path = "basketball\Butler_Jimmy"
    images_list = rename_and_load_images(folder_path)
    print(f"Loaded {len(images_list)} images.")
    # Display the first 10 images
    display_images(images=images_list, num_images=3)

    edges = edge_mask(images=images_list, line_size=5, blur_value=7)

    results = color_quantization(images=images_list, k=9)

    blurreds = noise_removal(result=results)

    cartoons_list=cartoonize(blurred=blurreds,edge=edges)
    print(f"Number of images cartoonized : {len(cartoons_list)}")
 
    # Display the first 10 images
    display_images(images=images_list,cartoons=cartoons_list, num_images=3)

    if save==True:
        save_cartoons(cartoons=cartoons_list, output_folder=folder_path+"/cartoon")
    

#%% Example usage

cartoonized_images = exec(save=False)