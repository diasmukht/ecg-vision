from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm
from django.contrib.auth.forms import AuthenticationForm

# === LANDING ===
def landing(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'analyzer/landing.html')

# === AUTHENTICATION ===

def user_login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        # Стандартная форма Django для входа
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            # Пытаемся найти пользователя
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
            # Создаем объект, но пока не сохраняем в БД
            user = form.save(commit=False)
            # Устанавливаем username равным email (лайфхак для Django Auth)
            user.username = form.cleaned_data['email'] 
            # Хешируем пароль (обязательно!)
            user.set_password(form.cleaned_data['password'])
            user.save()
            
            # Сразу логиним пользователя после регистрации
            login(request, user)
            return redirect('dashboard')
        else:
            # Выводим ошибки (например, пароли не совпали)
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{error}")

    return render(request, 'analyzer/auth/register.html')

def user_logout(request):
    logout(request)
    return redirect('landing')

# === DASHBOARD (Защищенные страницы) ===

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