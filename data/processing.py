import os

folder = 'C:/Users/TU/Desktop/Computer_Vision/Plate_Recognition/data/plate'
files = os.listdir(folder)

for i, name in enumerate(files):
    new_name = f'plate_0{i+1}.jpg'
    old_path = os.path.join(folder, name)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)