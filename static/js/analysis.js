const rawSignal = JSON.parse(document.getElementById('ecg-signal').textContent);
    const fs = 360; 
    const timeAxis = rawSignal.map((_, i) => i / fs);

    const trace = {
        x: timeAxis,
        y: rawSignal,
        mode: 'lines',
        line: { color: '#000000', width: 1.2 },
        name: 'ЭКГ'
    };

    const layout = {
        margin: { t: 30, r: 10, l: 40, b: 30 }, 
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {
            title: 'Время (сек)',
            gridcolor: '#ffcdd2',
            gridwidth: 1,
            dtick: 0.5, 
            zeroline: false,
            range: [0, 10], 
            showspikes: true
        },
        yaxis: {
            title: 'мВ',
            gridcolor: '#ffcdd2',
            gridwidth: 1,
            zeroline: false,
            fixedrange: false 
        },
        dragmode: 'pan',
        annotations: [] 
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    
    Plotly.newPlot('ecgChart', [trace], layout, config);

    
    const chartDiv = document.getElementById('ecgChart');

   
    chartDiv.on('plotly_click', function(data) {
      
        const xClick = data.points[0].x;
        const yClick = data.points[0].y;

       
        const noteText = prompt(`Добавить заметку на ${xClick.toFixed(2)} сек:\n(Оставьте пустым, чтобы отменить)`);

       
        if (noteText) {
           
            const newAnnotation = {
                x: xClick,               
                y: yClick,               
                xref: 'x',
                yref: 'y',
                text: noteText,         
                showarrow: true,         
                arrowhead: 2,            
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#0067FF',  
                ax: 0,                  
                ay: -45,                
                font: {
                    family: 'sans-serif',
                    size: 13,
                    color: '#ffffff'     
                },
                bgcolor: '#0067FF',      
                bordercolor: '#0067FF',
                borderwidth: 1,
                borderpad: 5,
                opacity: 0.9,
                bordercolor: '#ffffff'
            };

            
            const existingAnnotations = chartDiv.layout.annotations || [];
            existingAnnotations.push(newAnnotation);

        
            Plotly.relayout('ecgChart', { annotations: existingAnnotations });
        }
    });

    
    function zoomTo(startSec) {
        const start = Number(startSec);
        Plotly.relayout('ecgChart', {
            'xaxis.range': [start, start + 10]
        });
        
     
        const chartElement = document.getElementById('ecgChart');
        if (chartElement) {
            chartElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }