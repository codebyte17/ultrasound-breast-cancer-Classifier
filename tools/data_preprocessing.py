import numpy as np
from pathlib import Path
from PIL import Image


image_paths = Path("../data/raw/breast_ultrasound/Dataset_BUSI_with_GT/")
DER_PATH = "../data/proccessed/Overlay_dataset/"
target_path = Path(DER_PATH)

# Extract orignal images and ground truth
orignal_images_list = list(image_paths.rglob("?*).png"))
masked_images_list = list(image_paths.rglob("?*mask.png"))

# Transform the images by mapping the ground truth on orignal images and make the important location prominent
for idx, (og_image, msk_image) in enumerate(zip(orignal_images_list, masked_images_list)):
    try:
        # get class name
        class_name = og_image.parent.name
        if class_name == "normal":
            # Define the target path
            # save_path = target_path / class_name / f"{class_name}_{idx}.png"
            # shutil.copy2(str(og_image), str(save_path))
            continue

        # read the orignal and masked images
        img = np.asarray(Image.open(og_image))
        mask = np.asarray(Image.open(msk_image))

        overlay = img.copy()
        # Darken background
        overlay[~mask] = (overlay[~mask] * 0.2).astype(np.uint8)

        # Convert back to PIL Image
        output_image = Image.fromarray(overlay)

        # Save
        # Define the target path
        save_path = target_path / class_name / f"{class_name}_{idx}.png"
        output_image.save(save_path)

    except Exception as e:
        print(f"Image mismatched at index {idx}")
        # prints the exception type + message
        print(f"{type(e).__name__}: {e}")

