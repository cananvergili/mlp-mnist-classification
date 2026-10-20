import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms
from model import MLPModel

# 1. Modelin Başlatılması ve Ağırlıkların Yüklenmesi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel().to(device)

# Eğitim checkpoint dosyasından sadece model ağırlıklarını (state_dict) ayıklayarak yüklüyoruz
checkpoint = torch.load('training_results.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Modeli değerlendirme (evaluation) moduna al
model.eval()

# 2. Görüntü Ön İşleme (Preprocessing) Adımları
# Dışarıdan verilen fotoğrafları MNIST standartlarına (28x28 boyut, normalizasyon) uyarlar
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

okul_numarasi = ""
toplam_hane = 11

# Matplotlib figür ayarları: 11 haneli sayı için yan yana kutucuklar oluştur
fig, axes = plt.subplots(1, toplam_hane, figsize=(16, 3))
fig.suptitle("Okul Numarası Tahmini", fontsize=16)

# 3. Ardışık Rakam Okuma ve Tahmin Döngüsü
for i in range(1, toplam_hane + 1):
    dosya_adi = f"sayi{i}.jpeg"
    
    try:
        # Görüntüyü gri tonlamalı aç ve renkleri tersine çevir (Siyah arka plan, beyaz rakam)
        img = Image.open(dosya_adi).convert('L')
        img = ImageOps.invert(img)
        
        # Görüntüyü tensöre çevirip cihaza (CPU/GPU) aktar
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Gradyan hesaplamasını kapatarak modeli test et
        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1).item()
            
        okul_numarasi += str(prediction)
        
        # Modelin algıladığı görüntüyü ve tahminini ekranda göster
        axes[i-1].imshow(img, cmap='gray')
        axes[i-1].set_title(str(prediction))
        axes[i-1].axis('off')
        
    except FileNotFoundError:
        # Klasörde ilgili rakam fotoğrafı bulunamazsa uyarı ver
        print(f"Uyarı: {dosya_adi} bulunamadı!")
        axes[i-1].set_title("HATA")
        axes[i-1].axis('off')
        okul_numarasi += "?"

# 4. Sonuçların Yazdırılması ve Görselin Kaydedilmesi
print("\n" + "="*40)
print(f"MODELİN OKUDUĞU OKUL NUMARASI: {okul_numarasi}")
print("="*40 + "\n")

plt.tight_layout()
plt.savefig("okul_numarasi_sonuc.png")
plt.show()