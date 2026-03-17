from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
import datetime



class Doctor(AbstractUser):

    middle_name = models.CharField("Отчество", max_length=50, blank=True)
    phone = models.CharField("Телефон", max_length=20, blank=True)
    specialization = models.CharField("Специализация", max_length=100, default="Кардиолог")
    

    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)

    def get_full_name_rus(self):
        return f"{self.last_name} {self.first_name} {self.middle_name}".strip()

    class Meta:
        verbose_name = "Врач"
        verbose_name_plural = "Врачи"


class Patient(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, verbose_name="Лечащий врач")
    first_name = models.CharField("Имя", max_length=50)
    last_name = models.CharField("Фамилия", max_length=50)
    middle_name = models.CharField("Отчество", max_length=50, blank=True)
    birth_date = models.DateField("Дата рождения")
    gender = models.CharField("Пол", max_length=10, choices=[('M', 'Мужской'), ('F', 'Женский')])
    phone = models.CharField("Телефон", max_length=20, blank=True)
    email = models.EmailField("Email", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.last_name} {self.first_name}"

    @property
    def age(self):
        today = datetime.date.today()
        return today.year - self.birth_date.year - ((today.month, today.day) < (self.birth_date.month, self.birth_date.day))

    @property
    def full_name(self):
        return f"{self.last_name} {self.first_name} {self.middle_name}"


class ECGExamination(models.Model):

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='ecgs', verbose_name="Пациент")
    doctor = models.ForeignKey(Doctor, on_delete=models.SET_NULL, null=True, verbose_name="Врач диагностики")
    

    ecg_file = models.FileField(upload_to='ecg_files/%Y/%m/%d/', verbose_name="Файл ЭКГ (.csv)")
    created_at = models.DateTimeField("Дата загрузки", default=timezone.now)
    

    hr = models.IntegerField("ЧСС (уд/мин)", null=True, blank=True)
    qrs_duration = models.FloatField("QRS (сек)", null=True, blank=True) 
    p_duration = models.FloatField("P (сек)", null=True, blank=True)     
    pq_interval = models.FloatField("PQ (сек)", null=True, blank=True)   
    qt_interval = models.FloatField("QT (сек)", null=True, blank=True)   
    
    rhythm_type = models.CharField("Ритм", max_length=100, default="Синусовый правильный")
    conclusion = models.TextField("Заключение врача", blank=True)
    

    ai_report_json = models.JSONField("Отчет ИИ", null=True, blank=True)
 
    STATUS_CHOICES = [
        ('pending', 'В обработке'),
        ('healthy', 'Здоров'),
        ('pathology', 'Патология'),
        ('warning', 'Требует внимания'),
    ]
    status = models.CharField("Статус ИИ", max_length=20, choices=STATUS_CHOICES, default='pending')

    class Meta:
        ordering = ['-created_at']
        verbose_name = "ЭКГ Обследование"

    def __str__(self):
        return f"Запись #{self.id} - {self.patient.last_name}"
    


