import torch
import random
import json
from Evtx.Evtx import Evtx
from evtlog import EvtLog
from evtreceiver import EvtLogReceiver
import xmltodict
from autoencoder import Autoencoder
import torch.nn as nn


def validate_and_convert(data):
    feature_len = len(data[0])
    for row in data:
        if len(row) != feature_len:
            raise ValueError(f"Неконсинстентная длина признаков: {len(row)} != {feature_len}")
        for val in row:
            if not isinstance(val, (int, float)):
                raise ValueError(f"Найден нечисловой элемент: {val}")
        row[:] = [float(x) for x in row]
    return data


def normalize_data(data):
    n_features = len(data[0])
    min_vals = [float('inf')] * n_features
    max_vals = [float('-inf')] * n_features
    
    for row in data:
        for i in range(n_features):
            min_vals[i] = min(min_vals[i], row[i])
            max_vals[i] = max(max_vals[i], row[i])
    
    normalized_data = []
    for row in data:
        new_row = []
        for i in range(n_features):
            if max_vals[i] != min_vals[i]:
                norm_val = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                norm_val = 0.0
            new_row.append(norm_val)
        normalized_data.append(new_row)
    
    return normalized_data, min_vals, max_vals


def train_test_split_manual(data, train_ratio=0.8):
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    test_data = data[n_train:]
    return train_data, test_data


def to_tensor(data):
    return torch.tensor(data, dtype=torch.double)


if __name__ == "__main__":
    print("Чтение и обработка журналов событий...")
    evt_receiver = EvtLogReceiver(None, None)
    test_data = [event for event in evt_receiver.receive()]
    data = validate_and_convert(test_data)

    print("Обработка данных завершена.\nПодготовка данных к обучению...")
    
    normalized_data, min_vals, max_vals = normalize_data(data)

    train_data, test_data = train_test_split_manual(normalized_data)

    train_tensor = to_tensor(train_data)
    test_tensor = to_tensor(test_data)

    print("Обучение модели...")
    model = Autoencoder(input_dim=train_tensor.shape[1])
    model.fit(nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001), train_data, epochs=50, batch_size=128, verbose=True)
    mse, threshold = model.test(test_data)
    print(f"Threshold: {threshold}")
    anomalies = model.detect(mse, threshold)
    print(f"Anomalies:{anomalies}")
    model.save("autoencoder.pth")