const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('real-file');
    const fileText = document.getElementById('file-text');
    const fileSubtext = document.getElementById('file-subtext');
    const submitBtn = document.getElementById('submitBtn');

    function handleFileSelection(file) {
        if (!file) return;

        const fileName = file.name;
        if (!fileName.toLowerCase().endsWith('.csv')) {
            alert('Пожалуйста, загрузите файл в формате .CSV');
            fileInput.value = ''; 
            resetDropZone();
            return;
        }

        fileText.innerHTML = `<span style="color: var(--primary-blue); font-weight: 800;">Файл готов:</span> ${fileName}`;
        fileSubtext.innerText = `Размер: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        dropZone.classList.add('has-file');
    }

    function resetDropZone() {
        fileText.innerText = 'Нажмите или перетащите файл сюда';
        fileSubtext.innerText = 'Поддерживаемый формат: .CSV';
        dropZone.classList.remove('has-file');
    }

    fileInput.addEventListener('change', function() {
        handleFileSelection(this.files[0]);
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });


    dropZone.addEventListener('drop', function(e) {
        let dt = e.dataTransfer;
        let files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelection(files[0]);
        }
    });


    function validateAndSubmit() {
        if (!fileInput.files || fileInput.files.length === 0) {
            alert("Пожалуйста, выберите файл ЭКГ.");
            return false;
        }


        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <svg class="spinner" viewBox="0 0 50 50" style="width: 22px; height: 22px; vertical-align: middle; margin-right: 10px;">
                <circle cx="25" cy="25" r="20" fill="none" stroke="white" stroke-width="5" stroke-linecap="round"></circle>
            </svg>
            Обработка данных...
        `;
        return true; 
    }