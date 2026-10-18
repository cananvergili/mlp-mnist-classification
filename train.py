import torch
import torch.nn as nn
import torch.optim as optim
import time

# Özel veri yükleyicilerimizi ve MLP model mimarimizi içe aktarıyoruz
from dataset import train_loader, val_loader
from model import MLPModel

# 1. Model, Kayıp (Loss) Fonksiyonu ve Optimizasyon Algoritmasının Tanımlanması
model = MLPModel()
# Çok sınıflı (multi-class) sınıflandırma problemleri için standart olan Cross Entropy Loss kullanıyoruz
criterion = nn.CrossEntropyLoss() 
# Ağırlık güncellemeleri için adaptif öğrenme oranı sunan Adam optimizasyon algoritması
optimizer = optim.Adam(model.parameters(), lr=0.001) 

epochs = 5 # Modelin tüm eğitim veri setini kaç kez göreceği (Epoch sayısı)
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("Eğitim süreci (Training) başlatılıyor...\n")
start_time = time.time()

for epoch in range(epochs):
    # --- EĞİTİM (TRAIN) FAZI ---
    model.train() # Modeli eğitim moduna al (ağırlıkların güncellenmesine izin verir)
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()             # 1. Önceki iterasyondan kalan gradyanları (türevleri) sıfırla
        outputs = model(images)           # 2. İleri yayılım (Forward pass): Modelin tahminlerini al
        loss = criterion(outputs, labels) # 3. Kayıp hesaplaması (Loss computation): Tahmin ile gerçek etiket arasındaki fark
        loss.backward()                   # 4. Geri yayılım (Backpropagation): Hatayı ağdaki ağırlıklara dağıt
        optimizer.step()                  # 5. Ağırlıkları güncelle (Optimization step)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # --- DOĞRULAMA (VALIDATION) FAZI ---
    model.eval() # Modeli değerlendirme moduna al (ağırlık güncellemelerini dondur)
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Validation sırasında gradyan hesaplamasını kapat (bellek ve işlemci tasarrufu sağlar)
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Her epoch sonunda metrikleri ekrana yazdır
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} - Train Acc: %{train_acc:.2f} | Val Loss: {val_loss:.4f} - Val Acc: %{val_acc:.2f}")

end_time = time.time()
training_time = end_time - start_time

print(f"\nEğitim başarıyla tamamlandı!")
print(f"Toplam Eğitim Süresi: {training_time:.2f} saniye")

# Gelecekteki analizler ve grafik çizimleri için model metriklerini diske kaydet
torch.save({
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'model_state_dict': model.state_dict()
}, 'training_results.pth')