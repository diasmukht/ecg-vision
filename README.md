ecgvision
├─ analyzer          # ЭКГ талдауға жауапты қосымша.
│  ├─ ai_models      # Оқытылған нейрондық желінің салмақтары
│  │  └─ ecgvision_ai.pth
│  ├─ ai_module.py   # нейрондық желі архитектурасы, сигналдарды сүзу және ИИ талдау логикасы
│  ├─ forms.py
│  ├─ models.py
│  ├─ templates      # HTML шаблондары
│  │  └─ analyzer
│  │     ├─ auth
│  │     ├─ dashboard
│  │     └─ landing.html
│  ├─ urls.py
│  └─ views.py
├─ core              # Жобаның негізгі баптаулары
│   └─settings.py
├─ manage.py
└─ static # статикалық файлдар
   ├─ css
   └─ images
