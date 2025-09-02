import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# # Генерация синтетических данных: нормальные - кластер около [0,0], аномалии - далеко
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))  # 1000 нормальных точек, 2 признака
anomaly_data = np.random.uniform(-10, 10, (100, 2))  # 100 аномалий
data = np.vstack([normal_data, anomaly_data])
labels = np.hstack([np.zeros(1000), np.ones(100)])  # 0 - норм, 1 - аномалия (для теста)

# Только нормальные данные для обучения
train_data, _ = train_test_split(normal_data, test_size=0.2, random_state=42)
train_data = torch.tensor(train_data, dtype=torch.float32)

# Модель автоэнкодера


# Инициализация
model = Autoencoder(input_dim=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Тестирование на всех данных
test_data = torch.tensor(data, dtype=torch.float32)
model.eval()
with torch.no_grad():
    reconstructed = model(test_data)
    mse = torch.mean((test_data - reconstructed)**2, dim=1).numpy()

# Порог: средняя MSE на нормальных + 3 std
normal_mse = mse[:1000]
threshold = np.mean(normal_mse) + 3 * np.std(normal_mse)
print(f'Threshold: {threshold:.4f}')

# Детекция
predictions = (mse > threshold).astype(int)
accuracy = np.mean(predictions == labels)
print(f'Accuracy on test data: {accuracy:.2f}')

# Примеры
print('Sample MSE for normal:', mse[:5])
print('Sample MSE for anomalies:', mse[-5:])
