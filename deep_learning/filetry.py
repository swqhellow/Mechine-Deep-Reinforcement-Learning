import os
import random
import shutil
def split_dataset_recoginition(sources_path:str, target_path='dataset', split_radio=0.8)->None:
    """_summary_
    Args:
        sources_path (str): 
        the source of your data files and format must be:
        data/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                img2.jpg
                ...
        target_path (str):the result folder will be:
        dataset(or your-filename)/
            train/
                class1/
                    img1.jpg
                    ...
            test/
                class1/
                    img1.jpg
                    ...
        split_radio (int):0~1,default = 0.8 
    """    
    # Create a folder in the target directory
    os.makedirs(f'{target_path}/train',exist_ok=True)
    os.makedirs(f'{target_path}/test',exist_ok=True)
    # get the class name in the  
    classes = os.listdir(sources_path)
    for cls in classes:
        #split the train and test image
        fileincls = os.listdir(f'{sources_path}/{cls}')
        imgs = [file for file in fileincls if file.endswith('.jpg')]
        # Randomly shuffle files in the folder
        random.shuffle(imgs)
        # make classdir in the train and test
        os.makedirs(f'{target_path}/train/{cls}',exist_ok=True)
        os.makedirs(f'{target_path}/test/{cls}',exist_ok=True)
        # copy the imgs to the dataset/class/train and test file 
        for img in imgs[:int(len(imgs)*split_radio)]:
            shutil.copy(f'{sources_path}/{cls}/{img}',f'{target_path}/train/{cls}')
        for img in imgs[int(len(imgs)*split_radio):]:
            shutil.copy(f'{sources_path}/{cls}/{img}',f'{target_path}/test/{cls}')
sources_path= r'downloaded_images' 
target_path = r'Learning/data/'
split_dataset_recoginition(sources_path, target_path, 0.8)