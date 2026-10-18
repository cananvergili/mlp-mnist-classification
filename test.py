import torch
from dataset import test_loader
from model import MLPModel

# 1. Modeli Başlat ve Önceden Eğitilmiş Ağırlıkları Yükle
model = MLPModel()
data = torch.load('training_results.pth')
model.load_state_dict(data['model_state_dict'])

# 2. Değerlendirme Moduna Geçiş
model.eval() # Ağırlık güncellemelerini durdur, modeli test (inference) moduna al
correct = 0
total = 0

print("Test verisi üzerinde model değerlendirmesi (evaluation) yapılıyor...")

# 3. İleri Yayılım ve Doğruluk (Accuracy) Hesaplaması
with torch.no_grad(): # Test aşamasında gradient (türev) hesaplamasını devre dışı bırak
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # En yüksek olasılıklı sınıfı (argmax) tahmin olarak belirle
        total += labels.size(0) # Toplam örnek sayısını güncelle
        correct += (predicted == labels).sum().item() # Doğru tahmin edilenlerin sayısını topla

# 4. Genel Başarı Yüzdesini Hesapla ve Raporla
test_acc = 100 * correct / total
print(f"Test Accuracy (Test Başarısı): %{test_acc:.2f}")