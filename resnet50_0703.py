import os, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# ─ 사용자 정의 이미지 로더 ─

def pil_loader_with_rgb(path):
    img = Image.open(path)
    if img.mode in ('P', 'RGBA'):
        img = img.convert('RGB')
    return img

# ────────────설정───────────────────────────────

data_dir      = './model_2/train'  # 데이터 포버 (fresh, moldy 하위 클래스 포함)
num_classes   = 2                  # 클래스 수 (곰판이/신선)
num_epochs    = 100                # 최대 학습 epoch
batch_size = 4                     # 필수
torch.backends.cudnn.benchmark = True  # 추가 권장
learning_rate = 5e-5              # 초기 학습률 (처음부터 학습 시 더 작게)
patience      = 7                  # EarlyStopping: 협상 없는 epoch 수
min_delta     = 1e-4               # EarlyStopping: 최소 협상 포건

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA 사용 유무:", device)

# ─────────────데이터 전처리────────────────────────

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.05, 0.05, 0.05, 0.01),
    transforms.RandomPerspective(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ─────────────────데이터셋 로딩 및 분리───────────────────────

full_dataset = datasets.ImageFolder(data_dir, transform=train_transform, loader=pil_loader_with_rgb)
val_dataset  = datasets.ImageFolder(data_dir, transform=val_transform, loader=pil_loader_with_rgb)

val_size     = int(0.2 * len(full_dataset))
train_size   = len(full_dataset) - val_size
indices = torch.randperm(len(full_dataset))
train_indices = indices[:train_size]
val_indices   = indices[train_size:]

train_dataset = Subset(full_dataset, train_indices)
val_dataset   = Subset(val_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

# ────────────모델 구성: ResNet50 scratch 학습────────────────────

model = resnet50(weights=None)  # ResNet50 사용, 사전학입 없이 scratch 학습

# ─ 가운치 초기화 ─
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(initialize_weights)

# ─ FC 레이어 수정 (출력 클래스 수에 맞춰) ─
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)
model.to(device)

# ────────────학습 설정──────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# ────────────학습 루프────────────────

best_val_acc = 0.0
counter = 0
train_loss_list, val_loss_list, val_acc_list = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_loss_list.append(train_loss)

    # ─ 검지 ─
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"└ Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")

    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'best_model_123.pth')
        print("  ✔ 모델 저장됨 (val_acc 감소)")
    else:
        counter += 1
        print(f"  ✘ 감소 없음 → EarlyStopping 카운터: {counter}/{patience}")
        if counter >= patience:
            print("  ⚠ 감소 없어서 멈충")
            break

    scheduler.step()

# ────────────가장 높은 val acc 출력────────────────
print(f"\n🎯 최고 검지 정확도 (Best Validation Accuracy): {best_val_acc:.2f}%")

# ────────────전체 데이터로 미세조정────────────────
print("\n전체 데이터로 재학습 (fine-tuning) 시작")
full_dataset = datasets.ImageFolder(data_dir, transform=train_transform, loader=pil_loader_with_rgb)
full_loader  = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model.load_state_dict(torch.load('best_model_123.pth'))
model.train()

for epoch in range(3):
    running_loss = 0.0
    for inputs, labels in tqdm(full_loader, desc=f"[FineTune {epoch+1}]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"  ➡ Fine-tuning Epoch {epoch+1}: Loss {running_loss / len(full_loader):.4f}")

torch.save(model.state_dict(), 'best_model_final_0703.pth')
print("\n전체 데이터 기반 최종 모델 저장 완료")

# ────────────최종 성능 리포트 및 시각화──────────────────
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=datasets.ImageFolder(data_dir).classes))

# ─ 손스 및 정확도 그래프 시각화 ─
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend(); plt.grid(); plt.title("Loss per Epoch")
plt.show()

plt.plot(val_acc_list, label='Val Accuracy')
plt.legend(); plt.grid(); plt.title("Validation Accuracy")
plt.show()
