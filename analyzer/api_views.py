# api_views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from .models import Patient, ECGExamination
from .serializers import ECGExaminationSerializer
from .ai_module import ECGService 

class AnalyzeECGAPIView(APIView):

    permission_classes = [IsAuthenticated]

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = ECGExaminationSerializer(data=request.data)
        
        if serializer.is_valid():
            patient = serializer.validated_data['patient']
            if patient.doctor != request.user:
                return Response(
                    {"error": "У вас нет доступа к этому пациенту."}, 
                    status=status.HTTP_403_FORBIDDEN
                )
            
            exam = serializer.save(doctor=request.user, status='pending')
            
            try:
                analysis_result = ECGService.analyze_file(exam.ecg_file.path)
                
                metrics = analysis_result.get('metrics', {})
                exam.hr = metrics.get('hr', 0)
                exam.qrs_duration = metrics.get('qrs', 0)
                exam.qt_interval = metrics.get('qt', 0)
                exam.pq_interval = metrics.get('pq', 0)
                
                exam.ai_report_json = analysis_result.get('ai_report', [])
                exam.status = analysis_result.get('status', 'healthy')
                exam.save()
                
                response_serializer = ECGExaminationSerializer(exam)
                return Response(response_serializer.data, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                exam.status = 'error'
                exam.save()
                return Response(
                    {"error": f"Ошибка анализа ИИ: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)