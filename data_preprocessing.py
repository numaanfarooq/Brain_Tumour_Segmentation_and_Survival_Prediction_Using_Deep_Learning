import os
import numpy as np
import cv2
import nibabel as nib
from glob import glob
from tqdm import tqdm
from albumentations import Resize, Normalize
from logger import setup_logger
from exception import CustomException

# Setup logger
logger = setup_logger("log_process_____121212")

# Create a directory
def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")
        raise CustomException(f"Error creating directory {path}: {str(e)}")

# Load data from HGG and LGG directories
def load_data(path):
    try:
        train_images, mask_images = [], []
        hgg_path = os.path.join(path, 'Training', 'HGG')
        lgg_path = os.path.join(path, 'Training', 'LGG')
        hgg_patients = sorted(glob(os.path.join(hgg_path, "*")))
        lgg_patients = sorted(glob(os.path.join(lgg_path, "*")))
        all_patients = hgg_patients + lgg_patients
        logger.info(f"Found {len(all_patients)} training patients in total.")

        for patient_dir in all_patients:
            flair = glob(os.path.join(patient_dir, "*_flair.nii.gz"))
            t1 = glob(os.path.join(patient_dir, "*_t1.nii.gz"))
            t1ce = glob(os.path.join(patient_dir, "*_t1ce.nii.gz"))
            t2 = glob(os.path.join(patient_dir, "*_t2.nii.gz"))
            seg = glob(os.path.join(patient_dir, "*_seg.nii.gz"))

            if flair and t1 and t1ce and t2 and seg:
                train_images.append((flair[0], t1[0], t1ce[0], t2[0]))
                mask_images.append(seg[0])
            else:
                logger.warning(f"Missing modalities for patient: {os.path.basename(patient_dir)}")

        logger.info(f"Loaded {len(train_images)} images and {len(mask_images)} masks.")

        val_path = os.path.join(path, 'Validation')
        val_patients = sorted(glob(os.path.join(val_path, "*")))
        logger.info(f"Found {len(val_patients)} validation patients in total.")

        val_images, val_masks = [], []
        for patient_dir in val_patients:
            flair = glob(os.path.join(patient_dir, "*_flair.nii.gz"))
            t1 = glob(os.path.join(patient_dir, "*_t1.nii.gz"))
            t1ce = glob(os.path.join(patient_dir, "*_t1ce.nii.gz"))
            t2 = glob(os.path.join(patient_dir, "*_t2.nii.gz"))
            seg = glob(os.path.join(patient_dir, "*_seg.nii.gz"))

            if flair and t1 and t1ce and t2 and seg:
                val_images.append((flair[0], t1[0], t1ce[0], t2[0]))
                val_masks.append(seg[0])
            else:
                logger.warning(f"Missing modalities for validation patient: {os.path.basename(patient_dir)}")

        return (train_images, mask_images), (val_images, val_masks)

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise CustomException(f"Error loading data: {str(e)}")

# Read NIfTI images
def read_nii(file_path):
    try:
        return nib.load(file_path).get_fdata()
    except Exception as e:
        logger.error(f"Error reading NIfTI file {file_path}: {str(e)}")
        raise CustomException(f"Error reading NIfTI file {file_path}: {str(e)}")

# Preprocessing function
def preprocess_image(image, mask):
    try:
        # Normalize and resize (necessary preprocessing)
        aug = Normalize(mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0))
        normalized = aug(image=image, mask=mask)
        image = normalized["image"]
        mask = normalized["mask"]

        resize = Resize(512, 512)
        resized = resize(image=image, mask=mask)
        return resized["image"], resized["mask"]

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise CustomException(f"Error in preprocessing: {str(e)}")

# Save preprocessed data without augmentation
def save_preprocessed_data(images, masks, save_path):
    try:
        total_saved = 0

        for idx, (modalities, mask) in tqdm(enumerate(zip(images, masks)), total=len(images), desc="Processing Images"):
            patient_name = os.path.basename(os.path.dirname(modalities[0]))

            # Read image modalities and mask
            flair = read_nii(modalities[0])
            t1 = read_nii(modalities[1])
            t1ce = read_nii(modalities[2])
            t2 = read_nii(modalities[3])
            mask = read_nii(mask)

            # Stack modalities into a single multi-channel array
            image = np.stack([flair, t1, t1ce, t2], axis=-1)
            slice_idx = image.shape[2] // 2  # Middle slice
            image_slice = image[:, :, slice_idx, :]
            mask_slice = mask[:, :, slice_idx]

            # Preprocess the image and mask
            image_slice, mask_slice = preprocess_image(image_slice, mask_slice)

            # Save the preprocessed image and mask
            tmp_image_name = f"{patient_name}.png"
            tmp_mask_name = f"{patient_name}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            create_dir(os.path.dirname(image_path))
            create_dir(os.path.dirname(mask_path))

            success_image = cv2.imwrite(image_path, (image_slice * 255).astype(np.uint8))
            success_mask = cv2.imwrite(mask_path, (mask_slice * 255).astype(np.uint8))

            if success_image and success_mask:
                total_saved += 1
                logger.info(f"Saved: {image_path} and {mask_path}")
            else:
                logger.warning(f"Failed to save: {image_path} and/or {mask_path}")

        logger.info(f"Total images saved: {total_saved}")

    except Exception as e:
        logger.error(f"Error during data saving: {str(e)}")
        raise CustomException(f"Error during data saving: {str(e)}")

# Main workflow
if __name__ == "__main__":
    try:
        np.random.seed(42)

        # Load the data
        data_path = r'data/raw'
        (train_images, mask_images), (val_images, val_masks) = load_data(data_path)

        # Create directories to save the preprocessed data
        create_dir(os.path.join("new_data", "train", "image"))
        create_dir(os.path.join("new_data", "train", "mask"))
        create_dir(os.path.join("new_data", "val", "image"))
        create_dir(os.path.join("new_data", "val", "mask"))

        logger.info("Starting data processing...")

        # Process training data (without augmentation)
        save_preprocessed_data(train_images, mask_images, os.path.join("new_data", "train"))

        # Save validation data as .png files
        for idx, (modalities, mask) in tqdm(enumerate(zip(val_images, val_masks)), total=len(val_images), desc="Processing Validation Images"):
            patient_name = os.path.basename(os.path.dirname(modalities[0]))

            flair = read_nii(modalities[0])
            t1 = read_nii(modalities[1])
            t1ce = read_nii(modalities[2])
            t2 = read_nii(modalities[3])
            mask = read_nii(mask)

            # Save only one middle slice for the mask
            slice_idx = flair.shape[2] // 2
            mask_slice = mask[:, :, slice_idx]

            # Save the middle slice of the image modalities
            for modality, mod_name in zip([flair, t1, t1ce, t2], ["flair", "t1", "t1ce", "t2"]):
                tmp_name = f"{patient_name}_{mod_name}.png"
                save_path = os.path.join("new_data", "val", "image", tmp_name)
                cv2.imwrite(save_path, (modality[:, :, slice_idx] * 255).astype(np.uint8))

            # Save the mask
            mask_name = f"{patient_name}_mask.png"
            mask_path = os.path.join("new_data", "val", "mask", mask_name)
            cv2.imwrite(mask_path, (mask_slice * 255).astype(np.uint8))

        logger.info("Validation data saved successfully.")

    except Exception as e:
        logger.error(f"Error in the main workflow: {str(e)}")
        raise CustomException(f"Error in the main workflow: {str(e)}")
