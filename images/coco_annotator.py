import json
import os
import cv2

# Create an empty list to store the annotations
annotations = []

# Iterate over the images in the dataset
for i, image_name in enumerate(os.listdir("images")):
    # Load the image
    image_path = os.path.join("images", image_name)
    img = cv2.imread(image_path)
    
    # Extract the width and height of the image
    height, width, _ = img.shape
    
    # Annotate the image with a bounding box and label
    annotation = {
        "id": i,  # Use a unique identifier for the annotation
        "image_id": i,  # Use the same identifier for the image
        "category_id": 1,  # Assign a category ID to the object
        "bbox": [0, 0, width, height],  # Specify the bounding box in the format [x, y, width, height]
        "area": width * height,  # Calculate the area of the bounding box
        "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
    }
    annotations.append(annotation)

# Create the COCO JSON object
coco_data = {
    "info": {
        "description": "My COCO dataset",  # Add a description for the dataset
        "url": "",  # Add a URL for the dataset (optional)
        "version": "1.0",  # Set the version of the dataset
        "year": 2021,  # Set the year the dataset was created
        "contributor": "",  # Add the name of the contributor (optional)
        "date_created": "2022-01-01T00:00:00",  # Set the date the dataset was created
    },
    "licenses": [],  # Add a list of licenses for the images in the dataset (optional)
    "images": [
        {
            "id": i,  # Use the same identifier as the annotation
            "width": width,  # Set the width of the image
            "height": height,  # Set the height of the image
            "file_name": image_name,  # Set the file name of the image
            "license": 1,  # Set the license for the image (optional)
        }
        for i, image_name in enumerate(os.listdir("images"))
    ],
    "annotations": annotations,  # Add the list of annotations to the JSON object
    "categories": [{"id": 1, "name": "object"}],  # Add a list of categories for the objects in the dataset
}

# Save the COCO JSON object to a file
with open("coco.json", "w") as f:
    json.dump(coco_data, f)