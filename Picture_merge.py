import os
from PIL import Image

"""
# Example usage
directory1 = 'C:\\Users\\yaoca\\Desktop\\data\\105_Ball'  # The first folder path
directory2 = 'C:\\Users\\yaoca\\Desktop\\data\\GASF_118_ball'  # Second folder path
output_directory = 'feature_fusion_Inner'  # Output folder path
merge_images(directory1, directory2, output_directory)
"""

#You need the same number of images in both folders

import os
from PIL import Image

# Define two folder paths
folder1 = 'c:\\data\\234'  # The first folder path
folder2 = 'C:\\data\\GASF_images_234'  # Second folder path
output_folder = 'feature_fusion_Outer_234'  # Output folder path

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Gets a list of image files in folder 1
files1 = [f for f in sorted(os.listdir(folder1)) if f.endswith('.jpeg')]

# Gets a list of image files in folder 2
files2 = [f for f in sorted(os.listdir(folder2)) if f.endswith('.jpg')]

# Check whether the number of files matches
if len(files1) != len(files2):
    print("The number of files does not match")
else:
    # Traverse the image file
    for i in range(len(files1)):
        # Build file path
        file1 = files1[i]
        file2 = files2[i]
        image_path1 = os.path.join(folder1, file1)
        image_path2 = os.path.join(folder2, file2)

        # Open picture file
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        # Check that the picture sizes match
        if img1.size[0] != img2.size[0]:
            print("Picture size does not match")
            break

        # Create a new picture object and splice it
        new_img = Image.new('RGB', (img1.size[0], img1.size[1] + img2.size[1]))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, img1.size[1]))

        # Build the save path and file name
        output_file = f"output_{i+1}.jpg"
        output_path = os.path.join(output_folder, output_file)

        # 保存拼接后的图片
        new_img.save(output_path)

        print(f"The stitched image has been saved: {output_file}")

