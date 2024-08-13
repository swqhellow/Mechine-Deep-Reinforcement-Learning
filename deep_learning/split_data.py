import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    # Ensure the train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop over each class in the source directory
    for class_name in os.listdir(source_dir):
        # class is the foldername which is also the classes
        
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create corresponding train and test class directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all file names in the class directory
        file_names = os.listdir(class_dir)
        random.shuffle(file_names)

        # Split file names into train and test sets
        split_index = int(len(file_names) * split_ratio)
        train_files = file_names[:split_index]
        test_files = file_names[split_index:]

        # Move files to the train and test directories
        for file_name in train_files:
            src_file = os.path.join(class_dir, file_name)
            dest_file = os.path.join(train_class_dir, file_name)
            shutil.copy(src_file, dest_file)

        for file_name in test_files:
            src_file = os.path.join(class_dir, file_name)
            dest_file = os.path.join(test_class_dir, file_name)
            shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    source_directory = r"downloaded_images"
    train_directory = r"data\datasets\train"
    test_directory = r"data\datasets\test"
    split_ratio = 0.8

    split_dataset(source_directory, train_directory, test_directory, split_ratio)
