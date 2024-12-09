# U-Net 기반 GAN을 이용한 이미지 변환 프로젝트

## 프로젝트 개요
U-Net 기반의 Generator와 PatchGAN Discriminator를 결합하여 이미지를 변환하는 GAN 모델입니다. 본 프로젝트는 손상된 이미지를 복원하거나 특정 이미지를 원하는 스타일로 변환하기 위한 목적을 가지고 있습니다.

---

## 데이터 처리

### 데이터셋 구성
- **학습 데이터**: `train_input` (손상된 이미지), `train_gt` (복원된 이미지)
- **테스트 데이터**: `test_input`
- **출력 결과**: 각 에포크별로 생성된 이미지를 ZIP 파일로 저장합니다.

### 데이터 전처리
- 이미지를 읽어서 [0, 1] 범위로 정규화
- 이미지 크기 조정 필요 시 추가 변환 가능

```python
class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        )
```

---

## 모델 설계

### Generator (U-Net)
- 인코더-디코더 구조를 사용하여 이미지 세부 정보를 복원.
- 스킵 연결로 특징을 보존하며 디코딩 과정에서 활용.

### Discriminator (PatchGAN)
- 지역별로 진짜/가짜 이미지를 구분하며, GAN 훈련의 안정성을 향상.

```python
class UNet(nn.Module):
    def __init__(self):
        # U-Net 구조 정의

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        # PatchGAN 구조 정의
```

---

## 학습 과정

### 손실 함수
- **Generator Loss**: Adversarial Loss + Pixel Loss (MSE)
- **Discriminator Loss**: Binary Cross-Entropy Loss (BCE)

```python
adversarial_loss = nn.BCELoss()
pixel_loss = nn.MSELoss()
```

### 학습 루프
- 에포크마다 Generator와 Discriminator를 훈련.
- 테스트 데이터를 통해 생성된 이미지를 ZIP 파일로 저장.

```python
for epoch in range(epochs):
    # Generator 및 Discriminator 훈련
    # 결과 저장 및 ZIP 압축
```

---

## 실험 결과
- 각 에포크별 생성된 이미지 결과를 `result/epoch_{n}.zip`로 저장.
- 모델의 학습 상태를 `checkpoint.pth`로 저장.

### 예시 결과
1. **손상된 입력 이미지**
2. **생성된 출력 이미지**
3. **Ground Truth**

이미지 비교를 통해 모델 성능 확인 가능.

---

## 실행 방법

### 의존성 설치
```bash
pip install torch torchvision tqdm opencv-python scikit-image
```

### 학습 실행
1. 데이터 준비
    - 학습 데이터: `train_input`, `train_gt`
    - 테스트 데이터: `test_input`
2. 학습 시작
    ```bash
    python train.py
    ```

### 테스트 실행
- 학습된 모델로 테스트 결과 생성:
    ```bash
    python test.py
    ```

---

## 파일 구조
```
project/
|-- train_input/        # 손상된 학습 이미지
|-- train_gt/           # 복원된 학습 이미지
|-- test_input/         # 테스트 이미지
|-- result/             # 출력 이미지 및 ZIP 파일
|-- checkpoint.pth      # 학습된 모델 가중치
|-- train.py            # 학습 스크립트
|-- test.py             # 테스트 스크립트
|-- README.md           # 프로젝트 설명 파일
```

---

## 라이선스
본 프로젝트는 MIT 라이선스를 따릅니다.

