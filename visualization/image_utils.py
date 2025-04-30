import torch
import numpy as np

import os
import re
import jsonc as json
from PIL import Image


def img_list_to_pil(img_list, cond_image = None, seperation = 10):
    if cond_image is not None:
        img_list.append(cond_image)

    widths, heights = zip(*(i.size for i in img_list))
    total_width = sum(widths) + seperation * len(img_list)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in img_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + seperation

    return new_im


def grid_image_visualize(images, row_size):
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths) * row_size + 10 * (row_size - 1)
    max_height = max(heights) * ((len(images) + row_size - 1) // row_size)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    for i, im in enumerate(images):
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0] + 10
        if (i + 1) % row_size == 0:
            x_offset = 0
            y_offset += im.size[1]

    return new_im

def process_images(images, res=512):
    res_images = []
    for image in images:
        crop_size = min(image.size)
        
        left = (image.size[0] - crop_size) // 2
        top = (image.size[1] - crop_size) // 2
        right = (image.size[0] + crop_size) // 2
        bottom = (image.size[1] + crop_size) // 2

        image = image.crop((left, top, right, bottom))
        image = image.resize((res, res), Image.BILINEAR)
        res_images.append(image)
    return res_images

def sanitize_prompt(prompt: str, max_len: int = 50) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]+', '_', prompt)
    return sanitized[:max_len].strip("_")

def get_next_index(folder_path: str) -> int:
    if not os.path.exists(folder_path):
        return 0

    pattern = re.compile(r'.*_(\d+)\.(?:png|json)$')
    max_index = -1

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx

    return max_index + 1

def save_results(
    args,
    source_prompt: str,
    target_prompt: str,
    images: Image.Image,
    mask_path: str = None,
):
    src_name = sanitize_prompt(source_prompt)
    tgt_name = sanitize_prompt(target_prompt)
    folder_name = f"{src_name}#{tgt_name}"

    output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    next_idx = get_next_index(output_dir)

    concated_image = img_list_to_pil([images[0], images[-1]], cond_image=None, seperation=10)
    concated_image.save(os.path.join(output_dir, f"concat_{next_idx}.png"))
    images[0].save(os.path.join(output_dir, f"input_{next_idx}.png"))
    images[-1].save(os.path.join(output_dir, f"output_{next_idx}.png"))
    if mask_path is not None:
        args.mask = mask_path
    args_filename = f"args_{next_idx}.json"
    args_path = os.path.join(output_dir, args_filename)
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Saved image to {output_dir} and args to {args_path}")
