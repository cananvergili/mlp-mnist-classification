import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

# CSV formatındaki MNIST verisini PyTorch Dataset yapısına çeviren sınıf
class MNISTCsvDataset(Dataset):
    def __init__(self, csv_file):
        print(f"'{csv_file}' okunuyor, bu işlem birkaç saniye sürebilir...")
        
        # Veri setini pandas kullanarak belleğe al
        self.data = pd.read_csv(csv_file)
        
        # İlk sütun hedef etiketleri (0-9), kalan sütunlar 784 piksellik görüntü verisidir
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.astype(np.float32)
        
        # Min-Max Normalizasyonu: Piksel değerlerini [0, 1] aralığına ölçekle
        self.images = self.images / 255.0
        
        # Tensör boyutlandırması: (Batch Size, Channel, Height, Width) -> (N, 1, 28, 28)
        self.images = self.images.reshape(-1, 1, 28, 28)
        
        # Z-Score Normalizasyonu: MNIST veri setinin global ortalama ve standart sapmasına göre
        self.images = (self.images - 0.1307) / 0.3081

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Modelin işleyebilmesi için verileri PyTorch tensörlerine dönüştür
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

print("Veri setleri hazırlanıyor...")
full_train_dataset = MNISTCsvDataset('mnist_train.csv')
test_dataset = MNISTCsvDataset('mnist_test.csv')

# Eğitim setinin %15'ini hiperparametre optimizasyonu (Validation) için ayır
total_train = len(full_train_dataset)
val_size = int(total_train * 0.15)
train_size = total_train - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f"\nEğitim verisi sayısı: {len(train_dataset)}")
print(f"Doğrulama (Validation) verisi sayısı: {len(val_dataset)}")
print(f"Test verisi sayısı: {len(test_dataset)}")

# Verileri modele mini-batch'ler (64'lük paketler) halinde beslemek için DataLoader oluştur
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nDataLoader'lar hazır! İşlem tamam.")