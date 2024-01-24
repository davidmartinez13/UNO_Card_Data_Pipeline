import cv2
import os
import json
import random
import numpy as np
import datetime
import math

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
"change1.png":"change",
"change2.png":"change",
"change3.png":"change",
"plus2blue.png":"plus2",
"plus2green.png":"plus2",
"plus2red.png":"plus2",
"plus2yellow.png":"plus2",
"plus4.png":"plus4",
"plus41.png":"plus4",
"plus42.png":"plus4",
"plus43.png":"plus4",
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

def add_noise(image, intensity=random.randint(0, 50)):
    """Add random noise to the image."""
    noise = np.random.normal(0, intensity, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def pixelate(image, pixel_size=random.randint(1, 5)):
    """Pixelate the image."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def adjust_brightness_contrast(image, alpha=random.uniform(0.5, 1.0), beta=random.randint(0, 10)):
    """Adjust brightness and contrast of the image."""
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def dummy_transform(image):
    return image

def rotate_image_non_cropped(cv_image, angle=10):
    (h, w) = cv_image.shape[:2]
    h_offset = 0
    w_offset = 0
    # center = (w // 2 +w_offset, h // 2+ h_offset)
    center = (w // 2 +w_offset, h // 2+ h_offset)

    new_w = int(w * abs(math.sin(math.radians(angle))) + h * abs(math.cos(math.radians(angle))))
    new_h = int(w * abs(math.cos(math.radians(angle))) + h * abs(math.sin(math.radians(angle))))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(cv_image, M, (new_w, new_h))

    return rotated_image

def rotate_image(image, angle=random.randint(-15, 15)):
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT, borderValue=(0, 0, 0))
    return rotated_image

def apply_shear(image, shear_factor=random.uniform(-0.22, 0.22)):
    rows, cols, _ = image.shape
    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])

    # Apply shear transformation
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0))
    
    return sheared_image

def load_and_place_images(background_path, crop_folder, image_id, annotations, image_out, recording, show_bbox = False, single=True, show = False):
    
    # Read the background image using OpenCV
    background = cv2.imread(background_path)
    
    bg_height, bg_width, _ = background.shape
    img_scaling = 5
    # Create a new image to draw on
    background_upscaled = cv2.resize(background,(bg_width*img_scaling, bg_height*img_scaling))
    height, width, _ = background_upscaled.shape
    result_image = background_upscaled.copy()
    # List all the cropped images in the specified folder
    crop_images = [f for f in os.listdir(crop_folder) if f.endswith(".png")]
    # Shuffle the list for random selection
    random.shuffle(crop_images)
    transforms = [rotate_image,apply_shear, dummy_transform]
    modifications = [add_noise, pixelate, adjust_brightness_contrast, dummy_transform]
    offset_x = random.randint(0, 1000)
    offset_y = random.randint(0, 650)
    # Iterate through the first 6 cropped images

    for i in range(min(6, len(crop_images))):
        if single and i!=0:
            continue
        # Read and resize the cropped image to fit the background
        crop_path = os.path.join(crop_folder, crop_images[i])
        crop = cv2.imread(crop_path)
        crop = remove_border(crop)
        if single:
            scale = random.uniform(1.5, 4.5)
            crop = cv2.resize(crop,(int(crop.shape[1]*scale), int(crop.shape[0]*scale)))
        new_crop = transforms[random.randint(0, len(transforms)-1)](crop)
        new_crop = modifications[random.randint(0, len(modifications)-1)](new_crop)
        # new_crop = apply_shear(crop)
        crop_height, crop_width, _ = new_crop.shape

        # Calculate the position to place the cropped image in the result image
        row = i % 2
        col = i // 2
        
        x_spacing = 1.5
        y_spacing = 0.9
        if single:
            x_position = (result_image.shape[1] - crop.shape[1]) // 2
            y_position = (result_image.shape[0] - crop.shape[0]) // 2
        else:
            x_position = col * int(bg_width // x_spacing) + offset_x
            y_position = row * int(bg_height// y_spacing) + offset_y
        # Calculate the centroid of the bounding box
        centroid_x, centroid_y = calculate_centroid([x_position, y_position, x_position+crop_width, y_position+crop_height])

        # Draw a bounding box around the placed crop

        # Paste the cropped image onto the result image
        # result_image[y_position:y_position+crop_height, x_position:x_position+crop_width] = new_crop
        alpha = 1  # Adjust the blending strength
        result_image[y_position:y_position+new_crop.shape[0], x_position:x_position+new_crop.shape[1]] = cv2.addWeighted(
            result_image[y_position:y_position+new_crop.shape[0], x_position:x_position+new_crop.shape[1]],
            1 - alpha,
            new_crop,
            alpha,
            0
        )
        x_min , y_min = x_position, y_position
        if show_bbox:
            cv2.rectangle(result_image, (x_position, y_position), (x_position+crop_width, y_position+crop_height), (0, 255, 0), 3)
            label = os.path.splitext(crop_images[i])[0]
            cv2.putText(result_image, label, (x_position, y_position-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # bounding_box = [centroid_x, centroid_x, crop_width, crop_height]
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

    if show:
        cv2.imshow("Result Image", result_image)
        
    # Create img data
    now = str(datetime.datetime.now()).replace(':','').replace(' ','_').replace('-','_')
    image_name = now+'.jpg'

    if recording:
        cv2.imwrite(image_out+"/"+image_name, result_image)

    image_data = {
        "id": image_id,  # Use the same identifier as the annotation
        "width": width,  # Set the width of the image
        "height": height,  # Set the height of the image
        "file_name": image_name,  # Set the file name of the image
    }

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        quit()
    
    return annotations, image_data

if __name__ == "__main__":
    recording = True
    # Replace "background.jpg" with the path to your background image
    background_path = "./images/savedImage.jpg"

    # Replace "crop_folder" with the path to the folder containing cropped images
    crop_folder = "./images/cards"
    image_out = "./images/dataset"
    annot_out = "./annotations/UNO_dataset.json"
    annotations = []
    images = []
    single = True
    for image_id in range(1000):
        annotations, image_data = load_and_place_images(background_path, crop_folder, image_id, annotations, image_out, recording,single=single)
        single = not single
        images.append(image_data)
    print(len(annotations))
    print(len(images))

    coco_data = {
    "info": {
        "description": "UNO dataset",  # Add a description for the dataset
        "url": "",  # Add a URL for the dataset (optional)
        "version": "1.0",  # Set the version of the dataset
        "year": 2024,  # Set the year the dataset was created
        "contributor": "",  # Add the name of the contributor (optional)
        "date_created": "2024-01-01T00:00:00",  # Set the date the dataset was created
    },
    "licenses": [],  # Add a list of licenses for the images in the dataset (optional)
    "images": images,
    "annotations": annotations,  # Add the list of annotations to the JSON object
    "categories": categories,  # Add a list of categories for the objects in the dataset
    }
    if recording:
        # Save the COCO JSON object to a file
        with open(annot_out, "w") as f:
            json.dump(coco_data, f)
