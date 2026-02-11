from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import ECGUploadForm, UserRegistrationForm, DoctorProfileForm, PatientForm
from .models import Patient, ECGExamination
from django.contrib.auth.forms import AuthenticationForm
from .ai_module import ECGService
import pandas as pd
import json

# Landing
def landing(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'analyzer/landing.html')

# Auth 

def user_login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
    
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
          
            user = authenticate(username=username, password=password)
            if user is not None:
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

# Dashboard

@login_required(login_url='login')
def dashboard_main(request):
    return render(request, 'analyzer/dashboard/main.html')

@login_required(login_url='login')
def dashboard_archive(request):
    
    examinations = ECGExamination.objects.filter(doctor=request.user).order_by('-created_at')
    
    return render(request, 'analyzer/dashboard/archive.html', {'examinations': examinations})

@login_required(login_url='login')
def dashboard_check_ecg(request):
    return render(request, 'analyzer/dashboard/check_ecg.html')

@login_required(login_url='login')
def dashboard_patients(request):
    return render(request, 'analyzer/dashboard/patients.html')

@login_required(login_url='login')
def dashboard_settings(request):
    return render(request, 'analyzer/dashboard/settings.html')

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
            messages.error(request, "Ошибка при обновлении. Проверьте данные.")
    else:
        form = DoctorProfileForm(instance=user)

    return render(request, 'analyzer/dashboard/settings.html', {'form': form})


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


# 1. ОБНОВЛЯЕМ ЭТУ ФУНКЦИЮ (меняем redirect)
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
                # ... ТВОЙ КОД ЗАПУСКА ИИ (оставляем как есть) ...
                file_path = exam.ecg_file.path
                results = ECGService.analyze_file(file_path)
                
                # Сохранение метрик (оставляем как есть)
                metrics = results.get('metrics', {})
                exam.hr = metrics.get('hr')
                exam.p_duration = metrics.get('p_duration')
                exam.qrs_duration = metrics.get('qrs_duration')
                exam.pq_interval = metrics.get('pq_interval')
                exam.qt_interval = metrics.get('qt_interval')
                exam.ai_report_json = results.get('ai_report', [])
                risk = results.get('risk_score', 0)
                
                if risk > 50: exam.status = 'warning'
                elif risk > 0: exam.status = 'pathology'
                else: exam.status = 'healthy'

                exam.save()
                
                # !!! ИЗМЕНЕНИЕ ЗДЕСЬ !!!
                # Вместо messages.success и редиректа в архив -> идем на анализ
                return redirect('ecg_analysis', pk=exam.pk)

            except Exception as e:
                print(f"Ошибка: {e}")
                exam.status = 'error'
                exam.save()
                messages.error(request, "Ошибка обработки")
    else:
        form = ECGUploadForm(request.user)

    return render(request, 'analyzer/dashboard/check_ecg.html', {'form': form})


# 2. НОВАЯ ФУНКЦИЯ: СТРАНИЦА АНАЛИЗА (ГРАФИКИ)
@login_required(login_url='login')
def ecg_analysis_view(request, pk):
    exam = ECGExamination.objects.get(pk=pk)
    

    try:
        df = pd.read_csv(exam.ecg_file.path)

        signal_data = df.iloc[:21600, 0].tolist() 
    except:
        signal_data = []


    context = {
        'exam': exam,
        'patient': exam.patient,
        'doctor': exam.doctor,
        'signal_data': signal_data, 
        'ai_report': exam.ai_report_json or []
    }
    return render(request, 'analyzer/dashboard/analysis.html', context)


# 3. НОВАЯ ФУНКЦИЯ: ПЕЧАТНЫЙ ОТЧЕТ
@login_required(login_url='login')
def ecg_report_view(request, pk):
    exam = ECGExamination.objects.get(pk=pk)
    return render(request, 'analyzer/dashboard/report.html', {'exam': exam})