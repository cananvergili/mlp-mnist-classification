import torch
import matplotlib.pyplot as plt

# Eğitim aşamasında kaydedilen model metriklerini belleğe yükle
data = torch.load('training_results.pth')

train_losses = data['train_losses']
val_losses = data['val_losses']
train_accuracies = data['train_accuracies']
val_accuracies = data['val_accuracies']

epochs = range(1, len(train_losses) + 1)

# Grafik penceresinin boyutlarını belirle
plt.figure(figsize=(12, 5))

# 1. Alt Grafik: Eğitim ve Doğrulama Kayıp (Loss) Değerleri
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
plt.title('Eğitim ve Doğrulama Kaybı (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Alt Grafik: Eğitim ve Doğrulama Başarı (Accuracy) Oranları
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'r-o', label='Validation Accuracy')
plt.title('Eğitim ve Doğrulama Başarısı (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Grafikleri hizala, akademik rapor için diske kaydet ve ekranda göster
plt.tight_layout()
plt.savefig('grafikler.png') 
print("Eğitim metrikleri grafiği 'grafikler.png' olarak çalışma dizinine kaydedildi.")
plt.show()