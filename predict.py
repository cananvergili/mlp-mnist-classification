import torch
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
from model import MLPModel
import matplotlib.pyplot as plt

# 1. Modeli Başlat ve Eğitilmiş Ağırlıkları (Weights) Yükle
model = MLPModel()
data = torch.load('training_results.pth')
model.load_state_dict(data['model_state_dict'])
model.eval() # Ağırlık güncellemelerini kapat, test (inference) moduna geç

# 2. Gerçek Dünya Verisini (El Yazısı) Sisteme Yükle
resim_yolu = 'rakam.jpeg'
img = Image.open(resim_yolu).convert('L') # Görüntüyü gri tonlamalı (Grayscale) matrise çevir

# MNIST formatına uyum için renkleri tersine çevir (Siyah arka plan, beyaz rakam)
img = ImageOps.invert(img)

# --- GÖRÜNTÜ ÖN İŞLEME (PREPROCESSING) ADIMLARI ---

# A. Gürültü Azaltma (Gaussian Blur): Fotoğraftaki pürüzleri ve hafif gölgeleri temizle
img = img.filter(ImageFilter.GaussianBlur(radius=1))

# B. Eşikleme (Thresholding): Gri tonları net siyah veya beyaz piksellere ayır
# Yoğunluğu 80'in üzerinde olan pikselleri maksimum değere (255) çek
img = img.point(lambda p: 255 if p > 80 else 0)

# C. Morfolojik Genişletme (Dilation): Modelin algılayabilmesi için rakam çizgilerini kalınlaştır
img = img.filter(ImageFilter.MaxFilter(size=3))

# D. Otomatik Kırpma (Bounding Box): Sadece rakamın bulunduğu aktif pikselleri izole et
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

# E. Dolgu (Padding): MNIST veri dağılımına (distribution) benzemesi için rakamı merkeze oturt
w, h = img.size
max_dim = max(w, h)
padding_w = (max_dim - w) // 2 + int(max_dim * 0.2)
padding_h = (max_dim - h) // 2 + int(max_dim * 0.2)
img = ImageOps.expand(img, border=(padding_w, padding_h), fill='black')
# ---------------------------------------------------

# 3. Boyutlandırma ve Tensor Dönüşümü
transform = transforms.Compose([
    transforms.Resize((28, 28)), # Uzamsal boyutu 28x28 piksele indirge
    transforms.ToTensor(),       # [0, 1] aralığında PyTorch Tensörüne çevir
    transforms.Normalize((0.1307,), (0.3081,)) # Z-Score Normalizasyonu
])

# Tensör boyutunu (Batch, Channel, Height, Width) formatına uyarla
img_tensor = transform(img).unsqueeze(0)

# 4. İleri Yayılım (Forward Pass) ve Tahmin (Inference)
with torch.no_grad(): # Bellek tasarrufu için gradient hesaplamasını devre dışı bırak
    output = model(img_tensor)
    _, predicted = torch.max(output.data, 1) # En yüksek olasılıklı sınıfı (argmax) seç

tahmin_edilen_rakam = predicted.item()
print(f"\n---> Modelin Gerçek Dünya Verisi İçin Tahmini: {tahmin_edilen_rakam} <---")

# İşlemden geçmiş (preprocessed) görüntüyü ve tahmin sonucunu ekranda görselleştir
plt.imshow(img, cmap='gray')
plt.title(f"Modelin Tahmini: {tahmin_edilen_rakam}")
plt.axis('off')
plt.show()