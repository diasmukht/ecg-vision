from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm, DoctorProfileForm, PatientForm
from .models import Patient
from django.contrib.auth.forms import AuthenticationForm

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
    return render(request, 'analyzer/dashboard/archive.html')

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