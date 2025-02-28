from PIL import Image
import os

def optimize_images(input_dir, output_dir, max_width=1920, max_height=1080):
    """
    Optimizes image size and resolution for web viewing.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        max_width (int): Maximum width for the optimized images.
        max_height (int): Maximum height for the optimized images.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_path = os.path.join(root, file)
                optimized_path = os.path.join(output_dir, f"optimized_{file}")

                with Image.open(image_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height

                    if width > max_width or height > max_height:
                        if width > height:
                            new_width = max_width
                            new_height = int(max_width / aspect_ratio)
                        else:
                            new_height = max_height
                            new_width = int(max_height * aspect_ratio)

                        img = img.resize((new_width, new_height), Image.ANTIALIAS)

                    img.save(optimized_path, optimize=True, quality=85)

# Example usage:
input_directory = "path/to/your/images"
output_directory = "optimized_images"
optimize_images(input_directory, output_directory)