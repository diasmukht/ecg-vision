from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token 
from analyzer import api_views
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
    path('dashboard/patient/<int:pk>/', views.dashboard_patient_detail, name='patient_detail'),    
    path('dashboard/patient/<int:pk>/edit/', views.dashboard_patient_edit, name='patient_edit'),

    path('dashboard/settings/', views.dashboard_settings, name='settings'),


    path('dashboard/check-ecg/', views.dashboard_check_ecg, name='check_ecg'), 
    path('dashboard/analysis/<int:pk>/', views.ecg_analysis_view, name='ecg_analysis'), 
    path('dashboard/report/<int:pk>/', views.ecg_report_view, name='ecg_report'),
    
    # --- API ENDPOINTS ---
    path('api/token/', obtain_auth_token, name='api_token_auth'),
    path('api/ecg/analyze/', api_views.AnalyzeECGAPIView.as_view(), name='api_ecg_analyze'),  
]