import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(root, file))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 이미지 크기 통일
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

def calculate_mean_std_strong1(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    total_images = len(dataset)
    for i, images in enumerate(loader):
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.var(2).sum(0)  # 수정: std 대신 var 사용
        
        if (i + 1) % (total_images // 10) == 0 or i + 1 == total_images:
            print(f"진행 상황: {(i + 1) / total_images * 100:.1f}% ({i + 1}/{total_images})")
    
    mean /= total_images
    std = torch.sqrt(std / total_images)  # 수정: 평균 분산에서 표준편차 계산
    
    return mean, std

def calculate_mean_std_strong2(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    # 첫 번째와 두 번째 모멘트 초기화
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)
    pixel_count = 0
    
    total_images = len(dataset)
    for i, images in enumerate(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        # 채널별 합계 계산
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        
        # Welford's online algorithm 적용
        fst_moment = (pixel_count * fst_moment + sum_) / (pixel_count + nb_pixels)
        snd_moment = (pixel_count * snd_moment + sum_of_square) / (pixel_count + nb_pixels)
        pixel_count += nb_pixels
        
        if (i + 1) % (total_images // 10) == 0 or i + 1 == total_images:
            print(f"진행 상황: {(i + 1) / total_images * 100:.1f}% ({i + 1}/{total_images})")
    
    # 최종 평균과 표준편차 계산
    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment ** 2)
    
    return mean, std

def calculate_mean_std_weak1(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    mean = torch.zeros(3)
    var = torch.zeros(3)  # 수정: std 대신 var 사용
    total_images = 0
    
    for i, images in enumerate(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        batch_mean = images.mean(dim=[0, 2])
        batch_var = images.var(dim=[0, 2])  # 수정: 배치별 분산 계산
        
        mean = (total_images * mean + batch_samples * batch_mean) / (total_images + batch_samples)
        var = (total_images * var + batch_samples * batch_var) / (total_images + batch_samples)
        
        total_images += batch_samples
        
        if (i + 1) % (len(loader) // 10) == 0 or i + 1 == len(loader):
            print(f"진행 상황: {(i + 1) / len(loader) * 100:.1f}% ({total_images}/{len(dataset)})")
    
    std = torch.sqrt(var)  # 수정: 분산에서 표준편차 계산
    
    return mean, std

def calculate_mean_std_weak2(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    
    for i, images in enumerate(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        # 배치의 채널별 평균과 표준편차 계산
        images = images.view(b, c, -1)
        batch_mean = images.mean(dim=2)
        batch_std = images.std(dim=2)
        
        # 가중 평균으로 업데이트
        mean += batch_mean.sum(0)
        std += batch_std.sum(0)
        total_pixels += b
        
        if (i + 1) % (len(loader) // 10) == 0 or i + 1 == len(loader):
            print(f"진행 상황: {(i + 1) / len(loader) * 100:.1f}% ({total_pixels}/{len(dataset)})")
    
    mean /= total_pixels
    std /= total_pixels
    
    return mean, std

def main(folder_path, method, batch_size):
    dataset = ImageFolderDataset(folder_path)
    print(f"총 이미지 수: {len(dataset)}")
    print(f"선택된 방법: {method}")
    print("계산 시작...")

    if method == 'strong':
        mean, std = calculate_mean_std_strong(dataset)
    else:  # weak
        mean, std = calculate_mean_std_weak(dataset, batch_size)
    
    print(f"\n계산 완료!")
    print(f"평균 (RGB): {mean}")
    print(f"표준편차 (RGB): {std}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 폴더의 평균과 표준편차 계산")
    parser.add_argument("folder_path", type=str, help="이미지가 저장된 폴더 경로")
    parser.add_argument("--method", type=str, choices=['strong', 'weak'], default='strong', help="계산 방법 선택 (strong 또는 weak)")
    parser.add_argument("--batch_size", type=int, default=16, help="weak 방법 사용 시 배치 크기 (기본값: 16)")
    args = parser.parse_args()
    
    main(args.folder_path, args.method, args.batch_size)