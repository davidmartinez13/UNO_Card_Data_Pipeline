import cv2
import os
import random
import numpy as np
label_map = {
"0blue.png":"zero",
"0green.png":"zero",
"0red.png":"zero",
"0yellow.png":"zero",
"1blue.png":"one",
"1green.png":"one",
"1red.png":"one",
"1yellow.png":"one",
"2blue.png":"two",
"2green.png":"two",
"2red.png":"two",
"2yellow.png":"two",
"3blue.png":"three",
"3green.png":"three",
"3red.png":"three",
"3yellow.png":"three",
"4blue.png":"four",
"4green.png":"four",
"4red.png":"four",
"4yellow.png":"four",
"5blue.png":"five",
"5green.png":"five",
"5red.png":"five",
"5yellow.png":"five",
"6blue.png":"six",
"6green.png":"six",
"6red.png":"six",
"6yellow.png":"six",
"7blue.png":"seven",
"7green.png":"seven",
"7red.png":"seven",
"7yellow.png":"seven",
"8blue.png":"eight",
"8green.png":"eight",
"8red.png":"eight",
"8yellow.png":"eight",
"9blue.png":"nine",
"9green.png":"nine",
"9red.png":"nine",
"9yellow.png":"nine",
"blockblue.png":"block",
"blockgreen.png":"block",
"blockred.png":"block",
"blockyellow.png":"block",
"change.png":"change",
"plus2blue.png":"plus2",
"plus2green.png":"plus2",
"plus2red.png":"plus2",
"plus2yellow.png":"plus2",
"plus4.png":"plus4",
"reverseblue.png":"reverse",
"reversegreen.png":"reverse",
"reversered.png":"reverse",
"reverseyellow.png":"reverse"
}
category_ids = {
"one": 1,
"two": 2,
"three": 3,
"four": 4,
"five": 5,
"six": 6,
"seven": 7,
"eight": 8,
"nine": 9,
"zero": 10,
"plus2": 11,
"block": 12,
"reverse": 13,
"change": 14,
"plus4": 15
 }
categories = [{"id": iD, "name": name} for name, iD in category_ids.items()]

def remove_border(image, h = 6 , w = 6):
    # Crop the image based on the bounding box
    y,x,c = image.shape
    cropped_image = image[h:y-h, w:x-w]

    # Resize the cropped image to the original shape
    cropped_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))

    return cropped_image

def calculate_centroid(box):
    x = int((box[0] + box[2]) / 2)
    y = int((box[1] + box[3]) / 2)
    return x, y

def load_and_place_images(background_path, crop_folder, image_id, annotations):
    
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

    offset_x = random.randint(0, 1000)
    offset_y = random.randint(0, 650)
    # Iterate through the first 6 cropped images
    for i in range(min(6, len(crop_images))):
        # Read and resize the cropped image to fit the background
        crop_path = os.path.join(crop_folder, crop_images[i])
        crop = cv2.imread(crop_path)
        crop = remove_border(crop)
        crop_height, crop_width, _ = crop.shape

        # Calculate the position to place the cropped image in the result image
        row = i % 2
        col = i // 2
        
        x_spacing = 1.5
        y_spacing = 0.9
        x_position = col * int(bg_width // x_spacing) + offset_x
        y_position = row * int(bg_height// y_spacing) + offset_y
        # Calculate the centroid of the bounding box
        centroid_x, centroid_y = calculate_centroid([x_position, y_position, x_position+crop_width, y_position+crop_height])

        # Draw a bounding box around the placed crop

        # Paste the cropped image onto the result image
        result_image[y_position:y_position+crop_height, x_position:x_position+crop_width] = crop
        cv2.rectangle(result_image, (x_position, y_position), (x_position+crop_width, y_position+crop_height), (0, 255, 0), 3)
        label = os.path.splitext(crop_images[i])[0]
        cv2.putText(result_image, label, (x_position, y_position-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # bounding_box = [centroid_x, centroid_x, crop_width, crop_height]
        x_min , y_min = x_position, y_position
        cv2.circle(result_image, (x_min, y_min), 5, (0, 0, 255), -1)
        # Display the final result
        # Create annotation:
        annotation = {
        "id": image_id*6 + i,  # Use a unique identifier for the annotation
        "image_id": image_id,  # Use the same identifier for the image
        "category_id": category_ids[label_map[crop_images[i]]],  # Assign a category ID to the object
        "bbox": [x_min, y_min, crop_width, crop_height],  # Specify the bounding box in the format [x, y, width, height]
        "area": crop_width * crop_height,  # Calculate the area of the bounding box
        "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
        }
        annotations.append(annotation)
    cv2.imshow("Result Image", result_image)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        quit()
    
    return annotations

if __name__ == "__main__":

    # Replace "background.jpg" with the path to your background image
    background_path = "./images/savedImage.jpg"

    # Replace "crop_folder" with the path to the folder containing cropped images
    crop_folder = "./images/cards"
    annotations = []
    images = []
    for image_id in range(20):
        annotations = load_and_place_images(background_path, crop_folder, image_id, annotations)
    print(annotations)
