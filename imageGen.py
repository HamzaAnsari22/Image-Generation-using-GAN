from PIL import Image

def resize_image(input_path, output_path, new_size):
    # Open the image file
    original_image = Image.open(input_path)

    # Resize the image
    resized_image = original_image.resize(new_size)

    # Save the resized image
    resized_image.save(output_path)

# Example usage
input_path = "C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/static/generated20231114174355.png"
output_path = "C:/Users/ansar/Downloads/FYP Haris BUKC/Code Files GUI/static/generated20231114174355.png"
new_size = (500, 500)  # Specify the new size in pixels

resize_image(input_path, output_path, new_size)