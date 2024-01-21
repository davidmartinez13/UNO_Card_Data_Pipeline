import cv2
import os
import random
import numpy as np

def remove_border(image,h = 6 , w = 6):
    # Crop the image based on the bounding box
    y,x,c = image.shape
    cropped_image = image[h:y-h, w:x-w]

    # Resize the cropped image to the original shape
    cropped_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))

    return cropped_image

def load_and_place_images(background_path, crop_folder):
    # Read the background image using OpenCV
    background = cv2.imread(background_path)
    
    bg_height, bg_width, _ = background.shape
    img_scaling = 5
    # Create a new image to draw on
    result_image = cv2.resize(background,(bg_width*img_scaling, bg_height*img_scaling))

    # List all the cropped images in the specified folder
    crop_images = [f for f in os.listdir(crop_folder) if f.endswith(".png")]

    # Shuffle the list for random selection
    random.shuffle(crop_images)

    # Iterate through the first 6 cropped images
    for i in range(min(6, len(crop_images))):
        # Read and resize the cropped image to fit the background
        crop_path = os.path.join(crop_folder, crop_images[i])
        crop = cv2.imread(crop_path)
        crop = remove_border(crop)
        print(crop.shape)
        crop_height, crop_width, _ = crop.shape
        crop = cv2.resize(crop, (int(crop_width), int(crop_height)))

        # Calculate the position to place the cropped image in the result image
        row = i % 2
        col = i // 2
        offset_x = 200
        offset_y = 300
        x_spacing = 1.5
        y_spacing = 0.9
        x_position = col * int(bg_width // x_spacing) + offset_x
        y_position = row * int(bg_height// y_spacing) + offset_y

        # Paste the cropped image onto the result image
        result_image[y_position:y_position+crop.shape[0], x_position:x_position+crop.shape[1]] = crop

        # Display the final result
        cv2.imshow("Result Image", result_image)
        key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        quit()

if __name__ == "__main__":

    # Replace 'background.jpg' with the path to your background image
    background_path = "./images/savedImage.jpg"

    # Replace 'crop_folder' with the path to the folder containing cropped images
    crop_folder = "./images/cards"
    while True:
        load_and_place_images(background_path, crop_folder)

