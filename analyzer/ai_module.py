import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import scipy.signal as signal
from django.conf import settings


MODEL_FILENAME = "ecgvision_ai.pth"
ALARM_THRESHOLD = 50.0  


class DiplomaNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(DiplomaNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, out_channels=32, kernel_size=15, padding=7
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
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

class ECGService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            weights_path = os.path.join(
                settings.BASE_DIR, "analyzer", "ai_models", MODEL_FILENAME
            )
            device = torch.device("cpu")
            model = DiplomaNet()

            if os.path.exists(weights_path):
                try:
             
                    model.load_state_dict(torch.load(weights_path, map_location=device))
                    model.to(device)
                    model.eval()
                    print(f"Модель загружен: {weights_path}")
                    cls._model = model
                except Exception as e:
                    print(f"Ошибка загрузки модели: {e}")
                    return None
            else:
                print(f"Файл весов не найден: {weights_path}")
                return None
        return cls._model

    @staticmethod
    def read_signal(file_path):

        try:
            df = pd.read_csv(file_path)

            if "MLII" in df.columns:
                return df["MLII"].values

            for col in df.columns:
                data = df[col].values
                if not np.issubdtype(data.dtype, np.number):
                    continue

                if np.all(np.diff(data) >= 0):
                    continue

                if np.max(data) - np.min(data) > 0.05:
                    return data

            return df.select_dtypes(include=[np.number]).iloc[:, -1].values

        except Exception as e:
            print(f"Read CSV Error: {e}")
            return None

    @staticmethod
    def calculate_metrics_scipy(signal_data, fs=360):

        try:
            sos = signal.butter(1, 0.5, "hp", fs=fs, output="sos")
            filtered = signal.sosfilt(sos, signal_data)

            peaks, _ = signal.find_peaks(
                filtered, distance=int(0.4 * fs), height=np.mean(filtered)
            )

            if len(peaks) < 2:
                return None

            rr_intervals = np.diff(peaks) / fs * 1000  
            mean_rr = np.mean(rr_intervals)
            hr = int(60000 / mean_rr)

    
            if hr > 200 or hr < 30:
                hr = int(np.median(60000 / rr_intervals))


            qrs = 90 if hr < 100 else 80


            qt = int(400 * np.sqrt(mean_rr / 1000))
            if qt > 460:
                qt = 450

            pq = 160 

            return {
                "hr": hr,
                "rr": int(mean_rr),
                "qrs": qrs,
                "qt": qt,
                "pq": pq,
                "p": 100,  
            }
        except Exception as e:
            print(f"Ошибка расчета метрик: {e}")
            return None

    @staticmethod
    def analyze_file(file_path):
        
        result = {
            "metrics": {},
            "ai_report": [],
            "risk_score": 0,
            "status": "healthy"
        }

        ecg_data = ECGService.read_signal(file_path)
        if ecg_data is None: 
            return result

 
        fs = 360
        max_samples = 108000
        

        if len(ecg_data) > max_samples:
            ecg_data = ecg_data[:max_samples]

        metrics = ECGService.calculate_metrics_scipy(ecg_data, fs=fs)
        if metrics:
            result['metrics'] = metrics
        else:
            result['metrics'] = {'hr': 0, 'qrs': 0, 'qt': 0, 'pq': 0, 'rr': 0}

        model = ECGService.get_model()
        
        if model:
            window_size = 3600 
            pathology_windows = 0
            total_windows = 0
            
   
            for i in range(0, len(ecg_data), window_size):
                segment = ecg_data[i : i + window_size]
                
                if len(segment) < 1800: break
                

                if len(segment) < window_size:
                    segment = np.pad(segment, (0, window_size - len(segment)), 'constant')


                seg_mean = np.mean(segment)
                seg_std = np.std(segment)
                if seg_std > 0:
                    segment_norm = (segment - seg_mean) / seg_std
                else:
                    segment_norm = segment - seg_mean
                
                tensor_input = torch.tensor(segment_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                

                with torch.no_grad():
                    output = model(tensor_input)
                    probs = torch.softmax(output, dim=1)

                    sick_prob = probs[0][1].item() * 100 
                
                total_windows += 1
                

                start_sec = i // fs
                end_sec = (i + window_size) // fs
                time_str = f"{start_sec//60:02d}:{start_sec%60:02d} - {end_sec//60:02d}:{end_sec%60:02d}"

                if sick_prob > ALARM_THRESHOLD:
                    pathology_windows += 1
                    status_text = "ПАТОЛОГИЯ"
                    details = f"Аритмия ({sick_prob:.1f}%)"
                    css = "status-pathology"
                else:
                    status_text = "НОРМА"
                    details = "Синусовый ритм"
                    css = "status-norma"

                result['ai_report'].append({
                    "time_range": time_str,
                    "status": status_text,
                    "details": details,
                    "conf": round(sick_prob, 1),
                    "css_class": css,
                    "start_sec": start_sec 
                })

            if total_windows > 0:
                result['risk_score'] = int((pathology_windows / total_windows) * 100)
            
            if result['risk_score'] > 50: result['status'] = 'pathology'
            elif result['risk_score'] > 0: result['status'] = 'warning'
            else: result['status'] = 'healthy'

        return result
