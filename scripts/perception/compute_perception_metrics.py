from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
import os

import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPVisionModel
from PIL import Image
import requests
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA

### === helper functions === ###

def visualize_computed_area(original_image, non_transparent_mask, reconstructed_image):

    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Original Image with highlighted computed areas
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    masked_original = np.copy(original_image)
    masked_original[..., :3][~non_transparent_mask] = [255, 0, 0]  # Highlight non-computed areas in red
    plt.imshow(masked_original)
    plt.axis('off')
    
    # Reconstructed Image with highlighted computed areas
    plt.subplot(1, 3, 2)
    plt.title("Reconstructed Image")
    masked_reconstructed = np.copy(reconstructed_image)
    masked_reconstructed[..., :3][~non_transparent_mask] = [255, 0, 0]  # Highlight non-computed areas in red
    plt.imshow(masked_reconstructed)
    plt.axis('off')
    
    # Mask Visualization
    plt.subplot(1, 3, 3)
    plt.title("Computed Areas Mask")
    plt.imshow(non_transparent_mask, cmap='gray')
    plt.axis('off')
    
    plt.show()


def near_color(pixel, target_color, threshold=10):
    """Check if the pixel color is near the target color."""
    return all(abs(pixel[channel] - target_color[channel]) <= threshold for channel in range(3))

def remove_background(img, task_name):
    """Remove the background from an image."""    
    # Get the background color for the task
    if "geoclidean" in task_name:
        background_color = None
    elif "shapeworld" in task_name:
        background_color = (0, 0, 0)
    elif "nlvr" in task_name:
        background_color = (211, 211, 211)
    else:
        background_color = (255, 255, 255)

    # Ensure img is a PIL Image object
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    
    # Convert the image to RGBA if not already
    img = img.convert("RGBA")
    
    # Process the image to remove the background
    datas = img.getdata()
    newData = []
    for item in datas:
        # Make pixel transparent if it matches (or is near) the background color
        if background_color and near_color(item, background_color):
            newData.append(item[:3] + (0,))
        else:
            newData.append(item)
    
    img.putdata(newData)
    return img


### === pixel-based similarity metrics === ###
def compute_psnr_ssim_metrics(original_image, reconstructed_image, if_visualize=False):
    # if image mode is not RGBA, convert it to RGBA
    if original_image.mode != 'RGBA':
        original_image = original_image.convert("RGBA")
    if reconstructed_image.mode != 'RGBA':
        reconstructed_image = reconstructed_image.convert("RGBA")
    
    original_image = img_as_float(np.array(original_image))
    reconstructed_image = img_as_float(np.array(reconstructed_image))

    original_alpha = original_image[..., 3]
    reconstructed_alpha = reconstructed_image[..., 3]
    
    # Find pixels that are non-transparent in either image
    non_transparent_mask = (original_alpha > 0) | (reconstructed_alpha > 0)
    
    # # Visualize the computed areas
    if if_visualize:
        visualize_computed_area(original_image, non_transparent_mask, reconstructed_image)

    # Filter out pixels based on the mask. We're only interested in the RGB channels for metric computation
    original_non_transparent_indices = np.where(non_transparent_mask)
    reconstructed_non_transparent_indices = np.where(non_transparent_mask)
    
    original_image_filtered = original_image[original_non_transparent_indices][:, :3]
    reconstructed_image_filtered = reconstructed_image[reconstructed_non_transparent_indices][:, :3]
    
    original_image_reshaped = original_image_filtered.reshape(-1, 1)
    reconstructed_image_reshaped = reconstructed_image_filtered.reshape(-1, 1)

    # Compute PSNR value
    psnr_value = psnr(original_image_reshaped, reconstructed_image_reshaped, data_range=original_image_reshaped.max() - original_image_reshaped.min())
    
    # Compute SSIM value, treating the reshaped image as grayscale because SSIM requires 2D images at least
    ssim_value = ssim(original_image_reshaped, reconstructed_image_reshaped, data_range=original_image_reshaped.max() - original_image_reshaped.min(), channel_axis=-1)
    
    return psnr_value, ssim_value


### neural network-based similarity metrics ###
def compute_similarity(model, processor, img1, img2, model_type="clip", device='cuda'):

    with torch.no_grad():
        inputs1 = processor(images=img1, return_tensors="pt").to(device)
        img_feat1 = model(**inputs1).last_hidden_state
    
    with torch.no_grad():
        inputs2 = processor(images=img2, return_tensors="pt").to(device)
        img_feat2 = model(**inputs2).last_hidden_state

    # # # flatten and normalize
    img_feat1 = img_feat1.view(img_feat1.size(0), -1)
    img_feat2 = img_feat2.view(img_feat2.size(0), -1)

    img_feat1 = torch.nn.functional.normalize(img_feat1, dim=1)
    img_feat2 = torch.nn.functional.normalize(img_feat2, dim=1)

    sim = cosine_similarity(img_feat1, img_feat2)

    return sim.item()


from glob import glob
from collections import defaultdict
import json
from tqdm import tqdm

if __name__ == "__main__":
    
    def find_subdirs(directory):
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        return subdirs
    
    def load_nn_model_and_processor(model_type, device):
        if model_type == "dinov2":
            model_name = 'facebook/dinov2-base'
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
        elif model_type == "clip":
            model_name = "openai/clip-vit-large-patch14" # "openai/clip-vit-base-patch32"
            processor = AutoProcessor.from_pretrained(model_name)
            model = CLIPVisionModel.from_pretrained(model_name).to(device)
        return model, processor

    # input and output paths
    result_root = "results/perception"
    output_path = "results/perception/perception_score_results.json"

    # configs
    do_background_removal = False

    # neural models
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model_type = "dinov2"
    dino_model, dino_processor = load_nn_model_and_processor(model_type, device)
    model_type = "clip"
    clip_model, clip_processor = load_nn_model_and_processor(model_type, device)
    
    
    task_dirs = glob(os.path.join(result_root, "*"))
    if os.path.exists(output_path):
        print("output file already exists, loading the existing file...")
        all_sims = json.load(open(output_path, 'r'))
    else:
        all_sims = {}

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        if task_name in all_sims:
            print(f"task {task_name} already exists in the output file, skipping...")
            continue
        instance_dirs = find_subdirs(task_dir)
        sims = defaultdict(list)
        instance_ids = []
        print("working on task:", task_name)
        for instance_dir in tqdm(instance_dirs):
            original_image_path = os.path.join(task_dir, instance_dir, 'input_img.png')
            reconstructed_image_path = os.path.join(task_dir, instance_dir, 'output_visualizations/pred_all.png')

            original_image = Image.open(original_image_path)
            reconstructed_image = Image.open(reconstructed_image_path)

            if do_background_removal:
                original_image_rm_bk = remove_background(original_image, task_name)
                reconstructed_image_rm_bk = remove_background(reconstructed_image, task_name)
            else:
                original_image_rm_bk = original_image
                reconstructed_image_rm_bk = reconstructed_image

            # Compute the pixel-based metrics
            psnr_value, ssim_value = compute_psnr_ssim_metrics(original_image_rm_bk, reconstructed_image_rm_bk)

            # Compute model-based similarity
            original_image = original_image.convert("RGB")
            reconstructed_image = reconstructed_image.convert("RGB")
            dino_sim = compute_similarity(dino_model, dino_processor, original_image, reconstructed_image, model_type="dinov2", device=device)
            clip_sim = compute_similarity(clip_model, clip_processor, original_image, reconstructed_image, model_type="clip", device=device)

            # log
            sims['psnr'].append(psnr_value)
            sims['ssim'].append(ssim_value)
            sims['dino_sim'].append(dino_sim)
            sims['clip_sim'].append(clip_sim)
            instance_ids.append(os.path.basename(instance_dir))

            # import pdb; pdb.set_trace()
            # break

        all_sims[task_name] = {
            "scores": sims,
            "instance_ids": instance_ids
        }
        
    avg_sims = {}
    for task_name, task_scores in all_sims.items():
        avg_sims[task_name] = {}
        for score_name, score_list in task_scores['scores'].items():
            avg_sims[task_name][score_name] = np.mean(score_list)
    
    all_sims['AVG'] = avg_sims

    with open(output_path, 'w') as f:
        json.dump(all_sims, f, indent=4)