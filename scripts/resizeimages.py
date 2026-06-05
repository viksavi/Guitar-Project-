import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

# --- CONFIGURATION ---
SOURCE_DIR = r"../../../dataset/deux/frames/2/video_a_p_s"
OUTPUT_DIR = r"../../../dataset/deux/frames/2/video_a_p_s/resize"
RESIZE_FACTOR = 0.25  # 1/4th of the original size
# ---------------------

def resize_single_image(image_path, output_dir, factor):
    """Resizes a single image and saves it to the output directory."""
    try:
        with Image.open(image_path) as img:
            new_width = int(img.width * factor)
            new_height = int(img.height * factor)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            output_path = Path(output_dir) / image_path.name
            
            resized_img.save(output_path, quality=90)
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

def main():
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in source_path.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    if total_images == 0:
        print("No valid images found. Check your SOURCE_DIR path.")
        return

    print("Starting resizing using multi-core processing...")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(resize_single_image, img_path, output_path, RESIZE_FACTOR)
            for img_path in image_files
        ]
        
        for i, future in enumerate(futures):
            if (i + 1) % 500 == 0 or (i + 1) == total_images:
                print(f"Progress: {i + 1}/{total_images} images processed")

    print(f"All resized images saved in {output_path}")

if __name__ == "__main__":
    main()
