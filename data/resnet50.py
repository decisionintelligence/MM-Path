import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from collections import Counter
import pickle
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',type=str)
    parser.add_argument('--city', type=str)
    opt, unknown = parser.parse_known_args()
    return opt


def load_image_rgb(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def calculate_image_statistics(image_folder_path, batch_size=1000):
    image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    if not image_files:
        print("No images found in the specified folder.")
        return

    image_sizes = []
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                img = img.convert("RGB")
                image_sizes.append(img.size)
        except Exception as e:
            print(f"Error reading image {image_file}: {e}")
            continue

    size_counter = Counter(image_sizes)
    most_common_size = size_counter.most_common(1)[0][0]
    print(f"Most common image size: {most_common_size}")

    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    total_pixels = 0

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for image_file in batch_files:
            img_array = load_image_rgb(image_file)
            if img_array is not None:
                batch_images.append(img_array / 255.0)

        if batch_images:
            batch_images = np.stack(batch_images, axis=0)
            mean_sum += batch_images.mean(axis=(0, 1, 2)) * len(batch_images)
            std_sum += batch_images.std(axis=(0, 1, 2)) * len(batch_images)
            total_pixels += len(batch_images)

    if total_pixels == 0:
        print("No valid images found for computation.")
    else:
        mean = mean_sum / total_pixels
        std = std_sum / total_pixels
        return mean.tolist(),std.tolist()




if __name__ == "__main__":
    opt = parse_opt()
    mean,std = calculate_image_statistics(opt.file_path)

    transform = transforms.Compose([
        transforms.Resize((125, 125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])
    def extract_embedding(image_path):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)


        with torch.no_grad():
            embedding = model(image)
            embedding = embedding.squeeze().numpy()
        return embedding

    folder_path = opt.file_path
    embeddings = {}
    for filename in tqdm(list(os.listdir(folder_path))):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            embedding = extract_embedding(image_path)
            embeddings[filename] = embedding

    os.makedirs(f'data/{opt.city}/', exist_ok=True)
    pickle.dump(embeddings,open(f'data/{opt.city}/image/resnet50.pkl','wb'))