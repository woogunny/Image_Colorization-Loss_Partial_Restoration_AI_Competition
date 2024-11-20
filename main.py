from dataset import CustomDataset
from model import *
# from utils import utils

import random
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
import zipfile


CFG = {
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':16,
    'SEED':42
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main():
    print("Hello")
























    # 경로 설정
    origin_dir = './train_gt'  # 원본 이미지 폴더 경로
    damage_dir = './train_input'  # 손상된 이미지 폴더 경로
    test_dir = './test_input'     # test 이미지 폴더 경로

    # 데이터 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 데이터셋 및 DataLoader 생성
    dataset = CustomDataset(damage_dir=damage_dir, origin_dir=origin_dir, transform=transform)
    # 3. 데이터셋을 60:20:20 비율로 분할
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 4. DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=CFG['BATCH_SIZE'], shuffle=False)

    # 각 데이터셋 크기 출력
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    # dataloader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
    # print(type(dataset))
    # print(type(dataloader))
    # print(np.ndarray(dataloader))
    # train_features, train_labels = next(iter(dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")





















    # 모델 저장을 위한 디렉토리 생성
    model_save_dir = "./saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    best_loss = float("inf")
    lambda_pixel = 100  # 픽셀 손실에 대한 가중치
            
    # Generator와 Discriminator 초기화
    generator = UNetGenerator()
    generator = generator.to(device)
    discriminator = PatchGANDiscriminator()
    discriminator = discriminator.to(device)

    generator.apply(weights_init_normal).to(device)
    discriminator.apply(weights_init_normal).to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr = CFG["LEARNING_RATE"])
    optimizer_D = optim.Adam(discriminator.parameters(), lr = CFG["LEARNING_RATE"]) 
    
    
    # 학습 관련 설정

    train_losses, val_losses, val_ssims = [], [], []
    
    for epoch in range(1, CFG['EPOCHS'] + 1):
        # Training Loop
        generator.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            print(f"pred{pred_fake}")
            print(f"torch.ones_pred{torch.ones_like(pred_fake)}")
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            train_loss += loss_G.item()
        
        train_losses.append(train_loss / len(train_loader))

        # Validation Loop
        generator.eval()
        val_loss, val_ssim = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                real_A = batch['A'].to(device)
                real_B = batch['B'].to(device)
                
                fake_B = generator(real_A)
                loss_pixel = criterion_pixelwise(fake_B, real_B)
                val_loss += loss_pixel.item()
                
                # Calculate SSIM
                for j in range(real_B.size(0)):
                    ssim_value = ssim(
                        real_B[j].cpu().numpy().transpose(1, 2, 0), 
                        fake_B[j].cpu().numpy().transpose(1, 2, 0), 
                        data_range = 512,
                        channel_axis=2
                    )
                    val_ssim += ssim_value
            
            val_losses.append(val_loss / len(val_loader))
            val_ssims.append(val_ssim / len(val_loader.dataset))
        
        print(f"[Epoch {epoch}/{CFG['EPOCHS']}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val SSIM: {val_ssims[-1]:.4f}")

    # Test 데이터 평가
    test_loss, test_ssim = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            fake_B = generator(real_A)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            test_loss += loss_pixel.item()
            
            for j in range(real_B.size(0)):
                ssim_value = ssim(
                    real_B[j].cpu().numpy().transpose(1, 2, 0), 
                    fake_B[j].cpu().numpy().transpose(1, 2, 0), 
                    data_range = 512,
                    channel_axis=2
                )
                test_ssim += ssim_value
        
    print(f"Test Loss: {test_loss / len(test_loader):.4f} | Test SSIM: {test_ssim / len(test_loader.dataset):.4f}")

    # 학습 결과 시각화 및 저장
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, CFG['EPOCHS'] + 1)
    plt.figure(figsize=(12, 6))

    # Train and Validation Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    # Validation SSIM 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ssims, label='Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM')
    plt.legend()

    plt.tight_layout()

    # 그래프를 이미지 파일로 저장
    plot_path = os.path.join(output_dir, "training_results.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")











    # 저장할 디렉토리 설정
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 이미지 로드 및 전처리
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)  # 배치 차원을 추가합니다.
        return image

    # 모델 경로 설정
    generator_path = os.path.join(model_save_dir, "best_generator.pth")

    # 모델 로드 및 설정
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(generator_path, weights_only=True))
    model.eval()

    # 파일 리스트 불러오기
    test_images = sorted(os.listdir(test_dir))

    # 모든 테스트 이미지에 대해 추론 수행
    for image_name in test_images:
        test_image_path = os.path.join(test_dir, image_name)

        # 손상된 테스트 이미지 로드 및 전처리
        test_image = load_image(test_image_path).to(device)

        with torch.no_grad():
            # 모델로 예측
            pred_image = model(test_image)
            pred_image = pred_image.cpu().squeeze(0)  # 배치 차원 제거
            pred_image = pred_image * 0.5 + 0.5  # 역정규화
            pred_image = pred_image.numpy().transpose(1, 2, 0)  # HWC로 변경
            pred_image = (pred_image * 255).astype('uint8')  # 0-255 범위로 변환
            
            # 예측된 이미지를 실제 이미지와 같은 512x512로 리사이즈
            pred_image_resized = cv2.resize(pred_image, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 결과 이미지 저장
        output_path = os.path.join(submission_dir, image_name)
        cv2.imwrite(output_path, cv2.cvtColor(pred_image_resized, cv2.COLOR_RGB2BGR))    
        
    print(f"Saved all images")


















    # 저장된 결과 이미지를 ZIP 파일로 압축
    zip_filename = "submission.zip"
    with zipfile.ZipFile(zip_filename, 'w') as submission_zip:
        for image_name in test_images:
            image_path = os.path.join(submission_dir, image_name)
            submission_zip.write(image_path, arcname=image_name)

    print(f"All images saved in {zip_filename}")


if __name__ == "__main__":
    main()