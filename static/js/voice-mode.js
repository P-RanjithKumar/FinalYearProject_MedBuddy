class VoiceMode {
    constructor() {
        this.isActive = false;
        this.isListening = false;
        this.isSpeaking = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.currentChatId = null;
        this.initializeVoiceMode();
    }

    initializeVoiceMode() {
        if ('webkitSpeechRecognition' in window) {
            console.log('Speech recognition is supported');
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = true;

            this.setupEventListeners();
        } else {
            console.error('Speech recognition not supported');
        }
    }

    setupEventListeners() {
        const voiceButton = document.getElementById('voice-mode-toggle');
        console.log('Voice button found:', voiceButton); // Debug log
        
        voiceButton.addEventListener('click', () => {
            console.log('Voice button clicked'); // Debug log
            this.toggleVoiceMode()
        });

        this.recognition.onstart = () => {
            console.log('Recognition started'); // Debug log
            this.isListening = true;
            document.body.classList.add('voice-listening');
        };

        this.recognition.onresult = (event) => {
            console.log('Got speech result'); // Debug log
            const transcript = Array.from(event.results)
                .map(result => result[0].transcript)
                .join('');
            
            console.log('Transcript:', transcript); // Debug log   
            
            if (event.results[0].isFinal) {
                this.handleVoiceInput(transcript);
            }
        };

        this.recognition.onend = () => {
            this.isListening = false;
            document.body.classList.remove('voice-listening');
            if (this.isActive) {
                this.recognition.start();
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error); // Add error handler
        };
    

        // Listen for chat ID changes
        document.addEventListener('chatIdChanged', (e) => {
            this.currentChatId = e.detail.chatId;
        });
    }


    toggleVoiceMode() {
        this.isActive = !this.isActive;
        document.body.classList.toggle('voice-mode-active', this.isActive);
        
        if (this.isActive) {
            this.startListening();
            this.showNotification('Voice mode activated');
        } else {
            this.stopListening();
            this.showNotification('Voice mode deactivated');
        }
    }


    startListening() {
        if (this.recognition) {
            this.recognition.start();
        }
    }

    stopListening() {
        if (this.recognition) {
            this.recognition.stop();
        }
    }

    async handleVoiceInput(text) {
        if (!this.currentChatId) {
            // Create new chat if none exists
            try {
                const response = await fetch('/chat/new', {
                    method: 'POST'
                });
                const data = await response.json();
                this.currentChatId = data.chat_id;
            } catch (error) {
                console.error('Error creating new chat:', error);
                return;
            }
        }

        // Send message with voice mode flag
        try {
            // Show processing state
            document.body.classList.add('voice-processing');
            
            let response;
            try {
                response = await fetch(`/chat/${this.currentChatId}/message`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: text,
                        voice_mode: true
                    })
                });
            } catch (error) {
                console.error('Network error:', error);
                this.showNotification('Failed to connect - check internet connection');
                document.body.classList.remove('voice-processing');
                this.isActive = false;
                document.body.classList.remove('voice-mode-active');
                return;
            }

            // Handle streaming response
            const reader = response.body.getReader();
            let accumulatedResponse = '';

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const chunk = new TextDecoder().decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.trim()) {
                        const data = JSON.parse(line);
                        if (data.token) {
                            accumulatedResponse += data.token;
                            // Update UI with the token
                            updateChatUI(data.token);
                        }
                    }
                }
            }

            // Speak the complete response
            if (this.isActive) {
                this.speakResponse(accumulatedResponse);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            this.showNotification('Processing failed - please try again');
            document.body.classList.remove('voice-processing');
            this.isActive = false;
            document.body.classList.remove('voice-mode-active');
        } finally {
            document.body.classList.remove('voice-processing');
        }
    }


    speakResponse(text) {
        if (this.isActive && !this.isSpeaking) {
            this.isSpeaking = true;
            document.body.classList.add('voice-speaking');

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.onend = () => {
                this.isSpeaking = false;
                document.body.classList.remove('voice-speaking');
                this.startListening();
            };

            this.synthesis.speak(utterance);
        }
    }

    showNotification(message) {
        // Add your notification logic here
        console.log(message);
    }
}

// Initialize voice mode
document.addEventListener('DOMContentLoaded', () => {
    window.voiceMode = new VoiceMode();
});

// Modify your existing chat response handler to include voice
function handleChatResponse(response) {
    // Your existing chat display logic
    
    // Add voice response if voice mode is active
    if (window.voiceMode && window.voiceMode.isActive) {
        window.voiceMode.speakResponse(response);
    }
}

