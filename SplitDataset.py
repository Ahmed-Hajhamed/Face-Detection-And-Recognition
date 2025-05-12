import os
import shutil
import random

DATASET_DIR = "archive"  # contains s1 to s40
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
TEST_COUNT = 3  # number of random test images per subject

random.seed(42)  # for reproducibility

# Ensure output directories exist
for base in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(base, exist_ok=True)

for subject in sorted(os.listdir(DATASET_DIR)):
    subject_path = os.path.join(DATASET_DIR, subject)
    if not os.path.isdir(subject_path):
        continue

    images = sorted(os.listdir(subject_path))
    if len(images) < TEST_COUNT:
        raise ValueError(f"Not enough images in {subject} to split.")

    test_imgs = random.sample(images, TEST_COUNT)
    train_imgs = [img for img in images if img not in test_imgs]

    for img_list, target_base in [(train_imgs, TRAIN_DIR), (test_imgs, TEST_DIR)]:
        subject_target = os.path.join(target_base, subject)
        os.makedirs(subject_target, exist_ok=True)
        for img in img_list:
            shutil.copy(os.path.join(subject_path, img),
                        os.path.join(subject_target, img))

print("Randomized dataset split completed.")
