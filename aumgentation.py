import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# PATHS
# ----------------------------
input_folder = r"E:\SkinTone_Dataset_Google\mst-e_data\subject_16"   # existing folder
output_folder = r"E:\SkinTone_Dataset_Google\augmented_subject_16"   # new folder for augmented images
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# AUGMENTATION CONFIGURATION
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=25,          # random rotation
    width_shift_range=0.15,     # horizontal shift
    height_shift_range=0.15,    # vertical shift
    zoom_range=0.2,             # zoom
    brightness_range=[0.7, 1.3],# change brightness
    horizontal_flip=True,       # flip left-right
    fill_mode='nearest'         # fill missing pixels
)

# ----------------------------
# AUGMENT EACH IMAGE
# ----------------------------
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # skip non-image files

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    
    # Generate 5 augmented images per input image
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=output_folder,
                              save_prefix="aug", save_format="jpg"):
        i += 1
        if i >= 5:  # number of augmented images per original
            break

print("✅ Augmentation completed!")
print(f"Augmented images saved to: {output_folder}")

