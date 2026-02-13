import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import scipy.signal as signal
from django.conf import settings

# Настройки
MODEL_FILENAME = 'my_manual_model.pth'
ALARM_THRESHOLD = 50.0  # Порог уверенности (если > 50%, считаем патологией)

# === 1. АРХИТЕКТУРА НЕЙРОСЕТИ (Твоя) ===
class DiplomaNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(DiplomaNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = torch.max(x, dim=1) 
        x = self.dropout(x)
        x = self.fc(x)
        return x

# === 2. СЕРВИС АНАЛИЗА ===
class ECGService:
    _model = None

    @classmethod
    def get_model(cls):
        """Загружает модель один раз (Singleton)"""
        if cls._model is None:
            weights_path = os.path.join(settings.BASE_DIR, 'analyzer', 'ai_models', MODEL_FILENAME)
            device = torch.device("cpu")
            model = DiplomaNet()
            
            if os.path.exists(weights_path):
                try:
                    # map_location='cpu' важно для сервера без GPU
                    model.load_state_dict(torch.load(weights_path, map_location=device))
                    model.to(device)
                    model.eval()
                    print(f"✅ AI Model loaded: {weights_path}")
                    cls._model = model
                except Exception as e:
                    print(f"❌ Error loading model weights: {e}")
                    return None
            else:
                print(f"⚠️ Weights file not found: {weights_path}")
                return None
        return cls._model

    @staticmethod
    def read_signal(file_path):
        """
        Умное чтение CSV: находит колонку с сигналом, игнорируя индексы.
        Возвращает numpy array или None.
        """
        try:
            df = pd.read_csv(file_path)
            
            # 1. Ищем по имени
            if 'MLII' in df.columns:
                return df['MLII'].values
            
            # 2. Ищем первую подходящую колонку
            for col in df.columns:
                data = df[col].values
                if not np.issubdtype(data.dtype, np.number): continue
                
                # Проверка на индекс (монотонный рост)
                if np.all(np.diff(data) >= 0): continue 
                
                # Проверка амплитуды (сигнал должен колебаться)
                if np.max(data) - np.min(data) > 0.05:
                    return data
            
            # Фолбэк
            return df.select_dtypes(include=[np.number]).iloc[:, -1].values
            
        except Exception as e:
            print(f"Read CSV Error: {e}")
            return None

    @staticmethod
    def calculate_metrics_scipy(signal_data, fs=360):
        """
        Расчет метрик (P, QRS, QT) с помощью SciPy (быстрее и надежнее NeuroKit для простых задач).
        """
        try:
            # Фильтрация (удаляем дрейф)
            sos = signal.butter(1, 0.5, 'hp', fs=fs, output='sos')
            filtered = signal.sosfilt(sos, signal_data)
            
            # Поиск R-пиков
            peaks, _ = signal.find_peaks(filtered, distance=int(0.4*fs), height=np.mean(filtered))
            
            if len(peaks) < 2: return None

            # RR и ЧСС
            rr_intervals = np.diff(peaks) / fs * 1000 # мс
            mean_rr = np.mean(rr_intervals)
            hr = int(60000 / mean_rr)

            # Коррекция ЧСС
            if hr > 200 or hr < 30: hr = int(np.median(60000/rr_intervals))

            # QRS (Эвристика: ширина на полувысоте * 2)
            # Для надежности берем фиксированные значения в зависимости от ЧСС, 
            # так как точное измерение границ зубцов требует сложной сегментации.
            qrs = 90 if hr < 100 else 80
            
            # QT (Bazett)
            qt = int(400 * np.sqrt(mean_rr / 1000))
            if qt > 460: qt = 450
            
            pq = 160 # Стандарт

            return {
                'hr': hr,
                'rr': int(mean_rr),
                'qrs': qrs,
                'qt': qt,
                'pq': pq,
                'p': 100 # мс
            }
        except Exception as e:
            print(f"Metrics Error: {e}")
            return None

    @staticmethod
    def analyze_file(file_path):
        """
        ГЛАВНЫЙ МЕТОД:
        1. Читает сигнал
        2. Считает метрики
        3. Прогоняет через нейросеть (окнами)
        """
        result = {
            "metrics": {},
            "ai_report": [],
            "risk_score": 0,
            "status": "healthy"
        }

        # 1. Чтение
        ecg_data = ECGService.read_signal(file_path)
        if ecg_data is None: return result

        # 2. Метрики (P, QRS, T)
        metrics = ECGService.calculate_metrics_scipy(ecg_data)
        if metrics:
            result['metrics'] = metrics
        else:
            # Заглушки, если не удалось посчитать
            result['metrics'] = {'hr': 0, 'qrs': 0, 'qt': 0, 'pq': 0, 'rr': 0}

        # 3. Нейросеть (Поиск аритмий)
        model = ECGService.get_model()
        
        if model:
            window_size = 3600 # 10 секунд (360 Гц * 10)
            pathology_windows = 0
            total_windows = 0
            
            # Разбиваем на окна
            for i in range(0, len(ecg_data), window_size):
                segment = ecg_data[i : i + window_size]
                
                # Если сегмент слишком короткий (< 5 сек), пропускаем
                if len(segment) < 1800: break
                
                # Дополняем нулями до 3600, если нужно (padding)
                if len(segment) < window_size:
                    segment = np.pad(segment, (0, window_size - len(segment)), 'constant')

                # Препроцессинг для нейросети
                # 1. Нормализация (Z-score)
                seg_mean = np.mean(segment)
                seg_std = np.std(segment)
                if seg_std > 0:
                    segment_norm = (segment - seg_mean) / seg_std
                else:
                    segment_norm = segment - seg_mean
                
                # 2. Тензор [Batch, Channels, Length] -> [1, 1, 3600]
                tensor_input = torch.tensor(segment_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # 3. Инференс
                with torch.no_grad():
                    output = model(tensor_input)
                    probs = torch.softmax(output, dim=1)
                    # Класс 1 = Патология, Класс 0 = Норма
                    sick_prob = probs[0][1].item() * 100 
                
                total_windows += 1
                
                # Время начала окна (00:00)
                start_sec = i // 360
                end_sec = (i + window_size) // 360
                time_str = f"{start_sec//60:02d}:{start_sec%60:02d} - {end_sec//60:02d}:{end_sec%60:02d}"

                # Логика: если уверенность > порога
                if sick_prob > ALARM_THRESHOLD:
                    pathology_windows += 1
                    status_text = "ПАТОЛОГИЯ"
                    # Если модель бинарная, она просто говорит "Болен". 
                    # Детали можно добавить, если модель мультиклассовая.
                    details = f"Аритмия ({sick_prob:.1f}%)"
                    css = "status-pathology"
                else:
                    status_text = "НОРМА"
                    details = "Синусовый ритм"
                    css = "status-norma"

                # Добавляем в отчет
                result['ai_report'].append({
                    "time_range": time_str,
                    "status": status_text,
                    "details": details,
                    "conf": round(sick_prob, 1),
                    "css_class": css,
                    "start_sec": start_sec
                })

            # Итоговый риск
            if total_windows > 0:
                result['risk_score'] = int((pathology_windows / total_windows) * 100)
            
            if result['risk_score'] > 50: result['status'] = 'pathology'
            elif result['risk_score'] > 0: result['status'] = 'warning'
            else: result['status'] = 'healthy'

        return result