# Resize the image to square image while keeping the main content (adding padding instead of cropping)

import argparse
from PIL import Image

argparser = argparse.ArgumentParser(prog="Image resizer", description="resize image using padding")
argparser.add_argument("image_path", type=str, help="path to the image")
argparser.add_argument("-o", required=True, type=str, help="path to save the output file")
argparser.add_argument("--size", type=int, help="size = width = height", required=False, default=225)
args = argparser.parse_args()

img = Image.open(args.image_path)

new_img = Image.new("RGBA", (args.size, args.size), (255, 255, 255, 0))  # Create a square canvas with transparency

# Paste the original image centered on the square canvas
paste_x = (args.size - img.width) // 2
paste_y = (args.size - img.height) // 2

new_img.paste(img, (paste_x, paste_y))

# Resize the padded square image to size by size
resized_image_with_padding = new_img.resize((args.size, args.size))
resized_image_with_padding.save(args.o)
