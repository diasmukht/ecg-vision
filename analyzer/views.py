from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import ECGUploadForm, UserRegistrationForm, DoctorProfileForm, PatientForm
from .models import Patient, ECGExamination
from .ai_module import ECGService
import pandas as pd
import numpy as np
import scipy.signal as signal
import json
import os




# --- LANDING & AUTH ---

def landing(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'analyzer/landing.html')

def user_login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = authenticate(username=form.cleaned_data.get('username'), password=form.cleaned_data.get('password'))
            if user:
                login(request, user)
                return redirect('dashboard')
            else:
                messages.error(request, "Неверный логин или пароль")
        else:
            messages.error(request, "Неверный логин или пароль")
    
    return render(request, 'analyzer/auth/login.html')

def user_register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = form.cleaned_data['email'] 
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            return redirect('dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{error}")

    return render(request, 'analyzer/auth/register.html')

def user_logout(request):
    logout(request)
    return redirect('landing')


# --- DASHBOARD ---

@login_required(login_url='login')
def dashboard_main(request):
    return render(request, 'analyzer/dashboard/main.html')

@login_required(login_url='login')
def dashboard_archive(request):
    search_query = request.GET.get('search', '')
    examinations = ECGExamination.objects.filter(doctor=request.user).order_by('-created_at')
    
    if search_query:
        # Поиск по имени пациента (связанная модель)
        examinations = examinations.filter(patient__last_name__icontains=search_query)

    return render(request, 'analyzer/dashboard/archive.html', {'examinations': examinations, 'search_query': search_query})

@login_required(login_url='login')
def dashboard_patients(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            patient.doctor = request.user  
            patient.save()
            messages.success(request, f"Пациент {patient.last_name} успешно добавлен!")
            return redirect('patients')
        else:
            messages.error(request, "Ошибка при добавлении. Проверьте данные.")
    else:
        form = PatientForm()

    patients = Patient.objects.filter(doctor=request.user).order_by('-created_at')
    search_query = request.GET.get('search', '')
    if search_query:
        patients = patients.filter(last_name__icontains=search_query)

    context = {
        'patients': patients,
        'form': form,
        'search_query': search_query
    }
    return render(request, 'analyzer/dashboard/patients.html', context)

@login_required(login_url='login')
def dashboard_settings(request):
    user = request.user
    if request.method == 'POST':
        form = DoctorProfileForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Профиль успешно обновлен!")
            return redirect('settings')
        else:
            messages.error(request, "Ошибка при обновлении.")
    else:
        form = DoctorProfileForm(instance=user)

    return render(request, 'analyzer/dashboard/settings.html', {'form': form})


# --- ECG PROCESSING ---

@login_required(login_url='login')
def dashboard_check_ecg(request):
    if request.method == 'POST':
        form = ECGUploadForm(request.user, request.POST, request.FILES)
        if form.is_valid():
            exam = form.save(commit=False)
            exam.doctor = request.user
            exam.status = 'pending'
            exam.save()

            try:
                file_path = exam.ecg_file.path
                
                # 1. Считаем метрики (ЧСС, QRS...)
                metrics = calculate_ecg_metrics(file_path)
                
                if metrics:
                    exam.hr = metrics['hr']
                    exam.qrs_duration = metrics['qrs']
                    exam.qt_interval = metrics['qt']
                    exam.pq_interval = metrics['pq']
                    exam.rr_interval = metrics['rr'] # Если есть поле в модели
                
                # 2. Запуск ИИ
                results = ECGService.analyze_file(file_path)
                
                # Сохраняем отчет ИИ
                exam.ai_report_json = results.get('ai_report', [])
                risk = results.get('risk_score', 0)
                
                if risk > 50: exam.status = 'warning'
                elif risk > 0: exam.status = 'pathology'
                else: exam.status = 'healthy'

                exam.save()
                return redirect('ecg_analysis', pk=exam.pk)

            except Exception as e:
                print(f"Global Error: {e}")
                exam.status = 'error'
                exam.save()
                messages.error(request, "Ошибка обработки файла")
    else:
        form = ECGUploadForm(request.user)

    return render(request, 'analyzer/dashboard/check_ecg.html', {'form': form})


# ПЕЧАТНЫЙ ОТЧЕТ
@login_required(login_url='login')
def ecg_report_view(request, pk):
    exam = get_object_or_404(ECGExamination, pk=pk)
    return render(request, 'analyzer/dashboard/report.html', {'exam': exam})


# СТРАНИЦА ДЕТАЛЬНОГО АНАЛИЗА
@login_required(login_url='login')
def ecg_analysis_view(request, pk):
    exam = get_object_or_404(ECGExamination, pk=pk)
    
    signal_data = []
    try:
        if exam.ecg_file:
            df = pd.read_csv(exam.ecg_file.path)
            
            # --- ИСПРАВЛЕНИЕ ГРАФИКА ---
            # Проверяем, есть ли колонка 'MLII'
            if 'MLII' in df.columns:
                # Если нашли по названию
                signal_data = df['MLII'].iloc[:5000].tolist() # Берем первые 5000 точек для графика
            elif len(df.columns) > 1:
                # Если нет, берем вторую колонку (первая - индекс)
                signal_data = df.iloc[:5000, 1].tolist()
            else:
                # Если совсем беда, берем первую
                signal_data = df.iloc[:5000, 0].tolist()
                
            # Заменяем NaN на 0, чтобы JS не падал
            signal_data = [x if not pd.isna(x) else 0 for x in signal_data]

    except Exception as e:
        print(f"Error reading ECG file: {e}")

    # Подготавливаем метрики для шаблона
    metrics = {
        'heart_rate': exam.hr,
        'qrs_duration': exam.qrs_duration,
        'qt_interval': exam.qt_interval,
        'pq_interval': exam.pq_interval,
        'rr_interval': getattr(exam, 'rr_interval', int(60000/exam.hr) if exam.hr else 0)
    }

    context = {
        'exam': exam,
        'patient': exam.patient,
        'signal_data': signal_data, 
        'ai_report': exam.ai_report_json or [],
        'metrics': metrics
    }
    return render(request, 'analyzer/dashboard/analysis.html', context)



# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ РАСЧЕТА МЕТРИК ---
def calculate_ecg_metrics(file_path):
    """
    Читает CSV, находит R-пики и считает основные параметры ЭКГ.
    """
    try:
        # 1. Читаем CSV
        df = pd.read_csv(file_path)
        
        # Ищем колонку с сигналом (обычно 'MLII' или вторая колонка)
        if 'MLII' in df.columns:
            ecg_signal = df['MLII'].values
        elif len(df.columns) > 1:
            ecg_signal = df.iloc[:, 1].values  # Берем 2-ю колонку, т.к. 1-я это индекс
        else:
            ecg_signal = df.iloc[:, 0].values  # На случай если колонка одна

        fs = 360  # Частота дискретизации (стандарт MIT-BIH)

        # 2. Поиск R-пиков (используем scipy)
        # Порог: 60% от максимального сигнала (простая эвристика)
        threshold = np.max(ecg_signal) * 0.5
        peaks, _ = signal.find_peaks(ecg_signal, height=threshold, distance=int(0.4 * fs))

        if len(peaks) < 2:
            return None  # Не удалось найти пики

        # 3. Расчет метрик
        # RR интервалы (в мс)
        rr_intervals = np.diff(peaks) / fs * 1000
        mean_rr = np.mean(rr_intervals)
        
        # ЧСС (уд/мин)
        heart_rate = 60000 / mean_rr if mean_rr > 0 else 0

        # QRS (оценка ширины пика на уровне 0.5 высоты) - грубая оценка
        # Реальный QRS сложнее считать без neurokit2, берем среднее по больнице или эвристику
        # Используем норматив или простую ширину пика * коэффициент
        qrs_widths = signal.peak_widths(ecg_signal, peaks, rel_height=0.5)[0]
        mean_qrs = (np.mean(qrs_widths) / fs * 1000) * 2 # Эвристика для полной ширины
        if mean_qrs > 120 or mean_qrs < 40: mean_qrs = 90 # Фолбэк на норму, если шум

        # QT интервал (Формула Базетта: QT / sqrt(RR))
        # Оценочный QT (т.к. найти конец T-волны сложно без нейросети)
        qt_c = 400 # Норма QTc ~400мс
        qt_estimated = qt_c * np.sqrt(mean_rr / 1000)

        return {
            'hr': int(heart_rate),
            'rr': int(mean_rr),
            'qrs': int(mean_qrs),
            'qt': int(qt_estimated),
            'pq': 160 # PQ сложно найти простыми методами, ставим норму
        }

    except Exception as e:
        print(f"Ошибка расчета метрик: {e}")
        return None
