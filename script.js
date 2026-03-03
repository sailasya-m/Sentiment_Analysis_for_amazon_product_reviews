document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultCard = document.getElementById('result-card');
    const sentimentLabel = document.getElementById('sentiment-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        resultCard.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Sentiment';
        }
    });

    function displayResult(data) {
        const { sentiment, confidence } = data;
        const confPercent = Math.round(confidence * 100);
        
        resultCard.className = 'result-card ' + sentiment.toLowerCase();
        sentimentLabel.textContent = sentiment;
        confidenceText.textContent = `Confidence: ${confPercent}%`;
        
        resultCard.classList.remove('hidden');
        
        // Trigger reflow for animation
        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = `${confPercent}%`;
        }, 50);
    }
});
