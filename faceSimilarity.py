import cv2
import numpy as np
from PIL import Image
import numpy as np
import torch
from PIL import ImageOps
from PIL import Image
import json
from io import BytesIO

# import pydevd_pycharm
# pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

def img_to_tensor(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor

def img_to_np(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    image_np = np.array(image).astype(np.float32)
    return image_np

def img_to_mask(input):
    i = ImageOps.exif_transpose(input)
    image = i.convert("RGB")
    new_np = np.array(image).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return mask_tensor

def np_to_tensor(input):
    image = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    return tensor

def tensor_to_img(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    return img

def tensor_to_np(image):
    image = image[0]
    i = 255. * image.cpu().numpy()
    result = np.clip(i, 0, 255).astype(np.uint8)
    return result

def np_to_mask(input):
    new_np = input.astype(np.float32) / 255.0
    tensor = torch.from_numpy(new_np).permute(2, 0, 1)[0:1, :, :]
    return tensor

class SimilarityPM:
    @classmethod
    def INPUT_TYPES(s):
        return \
            {
                "required": {
                    "main_image": ("IMAGE",),
                    "compare_image": ("IMAGE",),                    
                },
            }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "similarity_compare_faceplusplus"
    CATEGORY = "facecompare"
    
    
    def similarity_compare_faceplusplus(self, main_image, compare_image):
        import requests
        url = "https://api-cn.faceplusplus.com/facepp/v3/compare"

        img_byte_arr = BytesIO()
        img_byte_arr1 = BytesIO()
        tensor_to_img(main_image).save(img_byte_arr, format='PNG')
        image_data1 = img_byte_arr.getvalue()
        tensor_to_img(compare_image).save(img_byte_arr1, format='PNG')
        image_data2 = img_byte_arr1.getvalue()

        files = {
            'image_file1': image_data1,
            'image_file2': image_data2
        }
        
        data = {
            'api_key': '',
            'api_secret':  ''
        }

        try:
            response = requests.post(url, files=files, data=data)
            result = json.loads(response.text)
            print(result)
            if result['confidence'] > 80:
                print('两张图片是同一人')
            else:
                print('两张图片不是同一人')
            score = str(round(result['confidence'], 2))   
            return (score,)
        except Exception as e:
            print('Error:', e)
            return ("0",)

        
    
NODE_CLASS_MAPPINGS = {
    "Face-similarity": SimilarityPM,    
}
NODE_DISPLAY_NAME_MAPPINGS = {   
    "Face-similarity": "Face-similarity",
}
