from django import forms
from .models import Doctor

class UserRegistrationForm(forms.ModelForm):
    # Добавляем кастомное поле для ФИО
    full_name = forms.CharField(
        label="ФИО", 
        widget=forms.TextInput(attrs={'placeholder': 'Иванов Иван Иванович'}),
        error_messages={'required': 'Пожалуйста, введите ФИО'}
    )
    password = forms.CharField(widget=forms.PasswordInput)
    password_confirm = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = Doctor
        # Убираем first_name, оставляем только email и пароль
        fields = ['email', 'password']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if Doctor.objects.filter(email=email).exists():
            raise forms.ValidationError("Этот Email уже зарегистрирован.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")

        if password != password_confirm:
            raise forms.ValidationError("Пароли не совпадают")
        return cleaned_data

    # Переопределяем сохранение, чтобы разбить ФИО
    def save(self, commit=True):
        user = super().save(commit=False)
        
        # Берем строку "Иванов Иван Иванович"
        full_name = self.cleaned_data.get('full_name', '').strip()
        parts = full_name.split()
        
        # Логика разбиения (стандарт ФИО: Фамилия Имя Отчество)
        if len(parts) > 0:
            user.last_name = parts[0]        # 1-е слово -> Фамилия
        if len(parts) > 1:
            user.first_name = parts[1]       # 2-е слово -> Имя
        if len(parts) > 2:
            user.middle_name = ' '.join(parts[2:]) # Остальное -> Отчество
            
        if commit:
            user.save()
        return user


class DoctorProfileForm(forms.ModelForm):
    full_name = forms.CharField(
        label="ФИО",
        widget=forms.TextInput(attrs={'placeholder': 'Фамилия Имя Отчество'}),
        required=True
    )

    class Meta:
        model = Doctor
        fields = ['full_name', 'email', 'phone', 'specialization', 'avatar']
        widgets = {
            'email': forms.EmailInput(attrs={'readonly': 'readonly'}), # Email лучше не менять, это логин
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Если профиль уже существует, склеиваем ФИО для отображения в поле
        if self.instance.pk:
            parts = [
                self.instance.last_name,
                self.instance.first_name,
                self.instance.middle_name
            ]
            self.initial['full_name'] = ' '.join(filter(None, parts))

    def save(self, commit=True):
        user = super().save(commit=False)
        
        # Разбиваем строку ФИО обратно на части
        full_name = self.cleaned_data.get('full_name', '').strip()
        parts = full_name.split()
        
        if len(parts) > 0: user.last_name = parts[0]
        if len(parts) > 1: user.first_name = parts[1]
        if len(parts) > 2: user.middle_name = ' '.join(parts[2:])
        else: user.middle_name = '' # Если стерли отчество

        if commit:
            user.save()
        return user