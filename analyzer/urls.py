from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),

    path('dashboard/', views.dashboard_main, name='dashboard'),
    path('dashboard/', views.dashboard_main, name='dashboard'),
    path('dashboard/archive/', views.dashboard_archive, name='archive'),       
    path('dashboard/check-ecg/', views.dashboard_check_ecg, name='check_ecg'), 
    path('dashboard/patients/', views.dashboard_patients, name='patients'),    
    path('dashboard/settings/', views.dashboard_settings, name='settings'),


    path('dashboard/check-ecg/', views.dashboard_check_ecg, name='check_ecg'), 
    path('dashboard/analysis/<int:pk>/', views.ecg_analysis_view, name='ecg_analysis'), # Графики
    path('dashboard/report/<int:pk>/', views.ecg_report_view, name='ecg_report'),       # Печатный бланк
]