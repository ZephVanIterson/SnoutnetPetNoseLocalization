import os
from PIL import Image

def check_corrupt_jpeg(directory):
    for filename in os.listdir(directory):
        try:
            with Image.open(os.path.join(directory, filename)) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f"Corrupt JPEG file: {filename}")

check_corrupt_jpeg("oxford-iiit-pet-noses/images-original/images")