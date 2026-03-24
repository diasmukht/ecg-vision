from rest_framework import serializers
from .models import ECGExamination, Patient

class ECGExaminationSerializer(serializers.ModelSerializer):

    patient_id = serializers.PrimaryKeyRelatedField(
        queryset=Patient.objects.all(), source='patient', write_only=True
    )
    
    class Meta:
        model = ECGExamination
        fields = [
            'id', 'patient_id', 'ecg_file', 'created_at', 
            'hr', 'qrs_duration', 'p_duration', 'pq_interval', 'qt_interval',
            'rhythm_type', 'status', 'conclusion', 'ai_report_json'
        ]
        read_only_fields = [
            'id', 'created_at', 'hr', 'qrs_duration', 'p_duration', 
            'pq_interval', 'qt_interval', 'rhythm_type', 'status', 
            'conclusion', 'ai_report_json'
        ]