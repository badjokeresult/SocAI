import torch
import random
import json
from Evtx.Evtx import Evtx
from evtlog import EvtLog
import xmltodict
from autoencoder import Autoencoder
import concurrent.futures
import torch.nn as nn

# 1. Проверка структуры и преобразование в float
def validate_and_convert(data):
    # Проверяем, что все списки имеют одинаковую длину
    feature_len = len(data[0])
    for row in data:
        if len(row) != feature_len:
            raise ValueError(f"Неконсинстентная длина признаков: {len(row)} != {feature_len}")
        for val in row:
            if not isinstance(val, (int, float)):
                raise ValueError(f"Найден нечисловой элемент: {val}")
        row[:] = [float(x) for x in row]  # Преобразуем в float
    return data

# 2. Нормализация (Min-Max Scaling)
def normalize_data(data):
    n_features = len(data[0])
    min_vals = [float('inf')] * n_features
    max_vals = [float('-inf')] * n_features
    
    # Находим min и max для каждого признака
    for row in data:
        for i in range(n_features):
            min_vals[i] = min(min_vals[i], row[i])
            max_vals[i] = max(max_vals[i], row[i])
    
    # Нормализация: (x - min) / (max - min)
    normalized_data = []
    for row in data:
        new_row = []
        for i in range(n_features):
            if max_vals[i] != min_vals[i]:  # Избегаем деления на 0
                norm_val = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                norm_val = 0.0  # Если min == max, ставим 0
            new_row.append(norm_val)
        normalized_data.append(new_row)
    
    return normalized_data, min_vals, max_vals

# 3. Разделение данных (80% train, 20% test)
def train_test_split_manual(data, train_ratio=0.8):
    random.shuffle(data)  # Перемешиваем данные
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    test_data = data[n_train:]
    return train_data, test_data

# 4. Преобразование в тензоры
def to_tensor(data):
    return torch.tensor(data, dtype=torch.double)


def read_evtx(path):
    evtx_records = []
    with Evtx(path) as log:
        for record in log.records():
            json = xmltodict.parse(record.xml())
            evtx_records.append(EvtLog(json["Event"]["System"]["EventID"]["#text"], json["Event"]["System"]["TimeCreated"]["@SystemTime"], json["Event"]["System"]["Channel"], json["Event"]["System"]["Computer"], json["Event"]["EventData"]["Data"]).get_values())
    return evtx_records


if __name__ == "__main__":
#     test_data = []
#     threads = []
#     paths = (r"C:\Users\obf344\Desktop\Security.evtx", r"C:\Users\obf344\Desktop\Microsoft-Windows-Sysmon%4Operational.evtx")

#     print("Чтение и обработка журналов событий...")
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         future_to_path = {executor.submit(read_evtx, path): path for path in paths}
#         for future in concurrent.futures.as_completed(future_to_path):
#             pth = future_to_path[future]
#             test_data.extend(future.result())

#     # Проверка и преобразование
#     data = validate_and_convert(test_data)

#     print("Обработка данных завершена.")
# # Нормализация
#     normalized_data, min_vals, max_vals = normalize_data(data)
#     #print("Нормализованные данные:", normalized_data)

# # Разделение
#     train_data, test_data = train_test_split_manual(normalized_data)
#     #print("Обучающие данные:", train_data)
#     #print("Тестовые данные:", test_data)

# # Преобразование в тензоры
#     train_tensor = to_tensor(train_data)
#     test_tensor = to_tensor(test_data)
# #print("Обучающий тензор:", train_tensor)
# #print("Тестовый тензор:", test_tensor)

# # Обучение модели
#     print("Обучение модели...")
#     model = Autoencoder(input_dim=train_tensor.shape[1])
#     model.fit(nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001), train_data, epochs=50, batch_size=128, verbose=True)
#     mse, threshold = model.test(test_data)
#     print(f"Threshold: {threshold}")
#     anomalies = model.detect(mse, threshold)
#     print(f"Anomalies:{anomalies}")
#     model.save("autoencoder.pth")
    import win32evtlog

    h = win32evtlog.OpenEventLog("pkobf059", "Application")
    records = win32evtlog.ReadEventLog(h, win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ, 0)
    print(f"""
            Reserver: {records[0].Reserved}
            RecordNumber: {records[0].RecordNumber}
            ClosingRecordNumber: {records[0].ClosingRecordNumber}
            ComputerName: {records[0].ComputerName}
            Data: {records[0].Data}
            EventCategory: {records[0].EventCategory}
            EventID: {records[0].EventID}
            EventType: {records[0].EventType}
            ReservedFlags: {records[0].ReservedFlags}
            Sid: {records[0].Sid}
            SourceName: {records[0].SourceName}
            StringInserts: {records[0].StringInserts}
            TimeGenerated: {records[0].TimeGenerated}
            TimeWritten: {records[0].TimeWritten}""")