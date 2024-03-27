import os
import random
import shutil

def split_dataset(dataset_path, train_ratio=0.8):
    # Create target folders for training and test sets
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Gets subfolders for all categories in a folder
    classes = os.listdir(dataset_path)

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Gets all image files under the current category
        images = [f for f in os.listdir(class_dir) if f.endswith('.jpeg') or f.endswith('.png')]

        # Shuffles the image list order
        random.shuffle(images)

        # Computes a split index that divides the training set and the test set
        split_index = int(len(images) * train_ratio)

        # Copy the image file to the corresponding training set or test set folder
        for i, image in enumerate(images):
            src_path = os.path.join(class_dir, image)
            if i < split_index:
                dst_path = os.path.join(train_dir, class_name, image)
            else:
                dst_path = os.path.join(test_dir, class_name, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

        print(f"Finished splitting dataset for class: {class_name}")

    print("Dataset splitting completed.")

#  Specify the data set path and call the function to partition
dataset_path = "data"  # Replace with your data set path
split_dataset(dataset_path, train_ratio=0.8)
