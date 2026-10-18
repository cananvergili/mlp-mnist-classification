import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Layer Perceptron (MLP) Mimarisi Sınıfı
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        
        # Girdi Katmanı (Input Layer): 28x28 piksellik MNIST görüntüleri (784 boyutlu vektör)
        # 1. Gizli Katman (Hidden Layer 1): 784 nörondan 256 nörona
        self.fc1 = nn.Linear(28 * 28, 256)
        
        # 2. Gizli Katman (Hidden Layer 2): 256 nörondan 128 nörona
        self.fc2 = nn.Linear(256, 128)
        
        # Çıktı Katmanı (Output Layer): 128 nörondan 10 nörona (0-9 arası sınıflar)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Matris formundaki 2D görüntüyü 1D vektör haline getir (Flattening işlemi)
        x = x.view(-1, 28 * 28)
        
        # İleri yayılım (Forward pass): Gizli katmanlarda ReLU aktivasyon fonksiyonu uygulanır
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Çıktı katmanı (Logits): Softmax uygulanmaz, PyTorch CrossEntropyLoss bunu kendi içinde hesaplar
        x = self.fc3(x)
        return x

# Modeli test etmek ve yapısını incelemek için bağımsız çalışma bloğu
if __name__ == "__main__":
    model = MLPModel()
    print("Model mimarisi başarıyla oluşturuldu!\n")
    print(model)