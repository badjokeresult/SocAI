# # Генерация синтетических данных: нормальные - кластер около [0,0], аномалии - далеко
# np.random.seed(42)
# normal_data = np.random.normal(0, 1, (1000, 2))  # 1000 нормальных точек, 2 признака
# anomaly_data = np.random.uniform(-10, 10, (100, 2))  # 100 аномалий
# data = np.vstack([normal_data, anomaly_data])
# labels = np.hstack([np.zeros(1000), np.ones(100)])  # 0 - норм, 1 - аномалия (для теста)

# # Только нормальные данные для обучения
# train_data, _ = train_test_split(normal_data, test_size=0.2, random_state=42)
# train_data = torch.tensor(train_data, dtype=torch.float32)

# Модель автоэнкодера

#from evtlog import EvtLog


if __name__ == "__main__":
    from hashlib import sha256
    
    class A:
        def __init__(self, a, b):
            self.a = a
            self.b = b
        
        def __eq__(self, value):
            if isinstance(value, type(self)):
                return self.a == value.a and self.b == value.b
            return False

    obj = A(1, "hello")
    print(hash(obj))
        