from PIL import Image
import os

def crop_image(input_image_path, output_folder):
    # Open the image
    image = Image.open(input_image_path)

    # Get the dimensions of the original image
    original_width, original_height = image.size

    # Define the dimensions of the grid
    grid_width = 5
    grid_height = 5

    # Calculate the width and height of each crop window
    crop_width = original_width // grid_width
    crop_height = original_height // grid_height

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through the grid and crop each section
    for row in range(grid_height):
        for col in range(grid_width):
            left = col * crop_width
            top = row * crop_height
            right = (col + 1) * crop_width
            bottom = (row + 1) * crop_height

            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))

            # Save the cropped image to the output folder
            output_path = os.path.join(output_folder, f"crop_{row}_{col}.png")
            cropped_image.save(output_path)

if __name__ == "__main__":
    # Replace 'input_image.jpg' with the path to your input image
    input_image_path = "images/UNOCardsClassic2.png"

    # Replace 'output_folder' with the desired output folder path
    output_folder = "images/cards2"

    crop_image(input_image_path, output_folder)
