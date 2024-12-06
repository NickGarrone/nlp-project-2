import os
import random
import string
import shutil

src_folder = "unseen_files"

def generate_random_name(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_random_subfolders(src_folder, num_subfolders=3):

    if not os.path.exists(src_folder):
        print(f"Error: Folder '{src_folder}' does not exist.")
        return

    for root, dirs, files in os.walk(src_folder):
        subfolder_paths = []
        
        for _ in range(num_subfolders):
            subfolder_name = generate_random_name()
            subfolder_path = os.path.join(root, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            subfolder_paths.append(subfolder_path)
        
        for file in files:
            file_path = os.path.join(root, file)
            random_subfolder = random.choice(subfolder_paths)
            shutil.move(file_path, os.path.join(random_subfolder, file))

        print(f"Successfully moved {len(files)} files in '{root}' into {num_subfolders} random subfolders.")
        
        dirs.clear()


num_subfolders = random.randint(1, 4)
create_random_subfolders(src_folder, num_subfolders)
for root, dirs, files in os.walk(src_folder):
    for directory in dirs:
        print(directory)
        if random.random() > 0.5:
            print(f"{src_folder}/{directory}")
            create_random_subfolders(f"{src_folder}/{directory}", random.randint(1, 4))


