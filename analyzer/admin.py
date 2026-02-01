from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Doctor, Patient, ECGExamination


@admin.register(Doctor)
class DoctorAdmin(UserAdmin):

    list_display = ('username', 'first_name', 'last_name', 'specialization', 'is_staff')

    fieldsets = UserAdmin.fieldsets + (
        ('Дополнительная информация', {'fields': ('middle_name', 'phone', 'specialization', 'avatar')}),
    )

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('last_name', 'first_name', 'birth_date', 'gender', 'doctor')
    search_fields = ('last_name', 'first_name', 'phone')
    list_filter = ('gender', 'doctor')

@admin.register(ECGExamination)
class ECGAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'created_at', 'status', 'doctor')
    list_filter = ('status', 'created_at')
    readonly_fields = ('created_at',) #