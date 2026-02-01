import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import neurokit2 as nk
from django.conf import settings

# Настройки
MODEL_FILENAME = 'my_manual_model.pth'
ALARM_THRESHOLD = 40.0  # % уверенности для тревоги

# === 1. ТВОЯ АРХИТЕКТУРА ===
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
        x, _ = torch.max(x, dim=1) # Твой Max Pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

# === 2. КЛАСС ПРЕДСКАЗАНИЯ (СЕРВИС) ===
class ECGService:
    _model = None  # Singleton: держим модель в памяти, чтобы не грузить каждый раз

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Путь к файлу весов
            weights_path = os.path.join(settings.BASE_DIR, 'analyzer', 'ai_models', MODEL_FILENAME)
            
            device = torch.device("cpu") # На сервере используем CPU
            model = DiplomaNet()
            
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.to(device)
                model.eval() # Режим предсказания
                print(f"✅ Модель загружена из {weights_path}")
                cls._model = model
            else:
                print(f"❌ Ошибка: Файл весов не найден: {weights_path}")
                return None
        return cls._model

    @staticmethod
    def analyze_file(file_path):
        """
        Главная функция: принимает путь к CSV, возвращает полный отчет (metrics + ai_json)
        """
        results = {
            "metrics": {},    # P, QRS, QT, ЧСС
            "ai_report": [],  # Список аномалий
            "risk_score": 0.0 # Общий риск
        }

        # 1. Читаем файл
        try:
            df = pd.read_csv(file_path)
            # Пытаемся найти колонку с сигналом (обычно 'MLII' или вторая колонка)
            if 'MLII' in df.columns:
                signal = df['MLII'].values
            else:
                signal = df.iloc[:, 1].values # Берем 2-ю колонку, если нет названия
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            return results

        # 2. Считаем метрики (ЧСС, QRS...) через NeuroKit2
        try:
            # Очистка сигнала
            clean_signal = nk.ecg_clean(signal, sampling_rate=360, method="neurokit")
            # Поиск пиков
            _, rpeaks = nk.ecg_peaks(clean_signal, sampling_rate=360)
            # Расчет P, QRS, T
            _, waves = nk.ecg_delineate(clean_signal, rpeaks, sampling_rate=360, method="dwt")
            
            # Собираем средние значения (в секундах)
            results['metrics']['hr'] = int(len(rpeaks['ECG_R_Peaks']) / (len(signal)/360) * 60)
            
            # Функция безопасного получения среднего (если волны не найдены, ставим 0)
            def get_avg_duration(start_list, end_list):
                try:
                    durations = np.array(end_list) - np.array(start_list)
                    # Делим на 360, чтобы получить секунды, и берем nanmean (игнор пустых)
                    return round(np.nanmean(durations) / 360, 3) 
                except:
                    return 0.0

            results['metrics']['p_duration'] = get_avg_duration(waves['ECG_P_Onsets'], waves['ECG_P_Offsets'])
            results['metrics']['qrs_duration'] = get_avg_duration(waves['ECG_R_Onsets'], waves['ECG_R_Offsets'])
            results['metrics']['qt_interval'] = get_avg_duration(waves['ECG_R_Onsets'], waves['ECG_T_Offsets']) # Грубая оценка QT
            results['metrics']['pq_interval'] = get_avg_duration(waves['ECG_P_Onsets'], waves['ECG_R_Onsets'])

        except Exception as e:
            print(f"Ошибка NeuroKit: {e}")
            # Если не вышло, ставим заглушки
            results['metrics'] = {'hr': 0, 'p_duration': 0, 'qrs_duration': 0, 'qt_interval': 0, 'pq_interval': 0}

        # 3. Анализ ИИ (по окнам 10 сек)
        model = ECGService.get_model()
        if model:
            window_size = 3600 # 10 сек
            pathology_count = 0
            total_windows = 0
            
            for start in range(0, len(signal), window_size):
                end = start + window_size
                if end > len(signal): break
                
                chunk = signal[start:end]
                
                # Нормализация
                mean = np.mean(chunk)
                std = np.std(chunk)
                if std > 0: chunk = (chunk - mean) / std
                else: chunk = chunk - mean
                
                # Тензор
                tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1)
                    sick_prob = probs[0][1].item() * 100
                
                total_windows += 1
                
                # Формируем запись для отчета
                time_str = f"{start//360//60:02d}:{(start//360)%60:02d}"
                
                if sick_prob > ALARM_THRESHOLD:
                    status = "pathology"
                    pathology_count += 1
                    # Добавляем в отчет ТОЛЬКО патологии, чтобы не забивать базу
                    results['ai_report'].append({
                        "time": time_str,
                        "status": "Патология",
                        "conf": round(sick_prob, 1)
                    })
            
            # Расчет риска
            if total_windows > 0:
                results['risk_score'] = round((pathology_count / total_windows) * 100, 2)
        
        return results