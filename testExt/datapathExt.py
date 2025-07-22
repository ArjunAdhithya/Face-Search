import os

folder = "./dataExtended"
image_paths = []
for root,dirs,files in os.walk(folder):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.abspath(os.path.join(root,file))
            image_paths.append(img_path)

#print(len(image_paths))