from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import ECGUploadForm, UserRegistrationForm, DoctorProfileForm, PatientForm
from .models import Patient, ECGExamination
from .ai_module import ECGService  # <-- Импортируем наш единый модуль
import pandas as pd
import numpy as np

# --- LANDING & AUTH (Оставляем как есть) ---

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

# --- DASHBOARD (Оставляем как есть) ---

@login_required(login_url='login')
def dashboard_main(request):
    return render(request, 'analyzer/dashboard/main.html')

@login_required(login_url='login')
def dashboard_archive(request):
    search_query = request.GET.get('search', '')
    examinations = ECGExamination.objects.filter(doctor=request.user).order_by('-created_at')
    if search_query:
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
            messages.error(request, "Ошибка при добавлении.")
    else:
        form = PatientForm()
    patients = Patient.objects.filter(doctor=request.user).order_by('-created_at')
    search_query = request.GET.get('search', '')
    if search_query:
        patients = patients.filter(last_name__icontains=search_query)
    context = {'patients': patients, 'form': form, 'search_query': search_query}
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

# --- ECG PROCESSING (ИСПРАВЛЕННАЯ ЧАСТЬ) ---

# 1. ЗАГРУЗКА И АНАЛИЗ
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
                # ВЫЗЫВАЕМ ЕДИНЫЙ СЕРВИС
                # Он делает ВСЁ: читает, считает метрики, запускает нейросеть
                analysis_result = ECGService.analyze_file(exam.ecg_file.path)
                
                # Извлекаем метрики
                metrics = analysis_result.get('metrics', {})
                exam.hr = metrics.get('hr', 0)
                exam.qrs_duration = metrics.get('qrs', 0)
                exam.qt_interval = metrics.get('qt', 0)
                exam.pq_interval = metrics.get('pq', 0)
                # Если в модели есть rr_interval, раскомментируй:
                # exam.rr_interval = metrics.get('rr', 0)

                # Извлекаем отчет ИИ и статус
                exam.ai_report_json = analysis_result.get('ai_report', [])
                exam.status = analysis_result.get('status', 'healthy')
                
                exam.save()
                return redirect('ecg_analysis', pk=exam.pk)

            except Exception as e:
                print(f"Analysis Error: {e}")
                exam.status = 'error'
                exam.save()
                messages.error(request, "Ошибка при анализе файла")
        else:
             messages.error(request, "Ошибка формы")
    else:
        form = ECGUploadForm(request.user)

    return render(request, 'analyzer/dashboard/check_ecg.html', {'form': form})

# 2. ПРОСМОТР РЕЗУЛЬТАТОВ
@login_required(login_url='login')
def ecg_analysis_view(request, pk):
    exam = get_object_or_404(ECGExamination, pk=pk)
    
    # Подготовка сигнала для графика (используем метод из сервиса, чтобы не дублировать код)
    signal_data = []
    try:
        # Читаем сигнал тем же умным способом, что и сервис
        raw_data = ECGService.read_signal(exam.ecg_file.path)
        if raw_data is not None:
            # Берем первые 30 секунд для JS (360 * 30 = 10800)
            signal_data = raw_data[:10800].tolist()
            # Заменяем NaN на 0
            signal_data = [x if not pd.isna(x) else 0 for x in signal_data]
    except Exception as e:
        print(f"Error reading signal for view: {e}")

    # Подготовка метрик для шаблона (форматирование в секунды)
    metrics_display = {
        'hr': exam.hr,
        'qrs': f"{(exam.qrs_duration or 0)/1000:.2f}", # мс -> с
        'qt': f"{(exam.qt_interval or 0)/1000:.2f}",
        'pq': f"{(exam.pq_interval or 0)/1000:.2f}",
        'p': "0.10" # Заглушка, т.к. P редко считают точно
    }

    # Сегменты берем ПРЯМО из JSON (который сгенерировал ИИ)
    segments = exam.ai_report_json or []

    context = {
        'exam': exam,
        'patient': exam.patient,
        'doctor': exam.doctor,
        'signal_data': signal_data,
        'metrics': metrics_display,
        'segments': segments
    }
    return render(request, 'analyzer/dashboard/analysis.html', context)

# ПЕЧАТНЫЙ ОТЧЕТ
@login_required(login_url='login')
def ecg_report_view(request, pk):
    exam = get_object_or_404(ECGExamination, pk=pk)
    return render(request, 'analyzer/dashboard/report.html', {'exam': exam})