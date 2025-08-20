/**
 * Step 5: Attendance Flow JavaScript
 * Handles real-time communication with Step 5 integrated system
 */

class Step5AttendanceFlow {
    constructor(config = {}) {
        this.config = {
            socketUrl: config.socketUrl || window.location.origin,
            videoElementId: config.videoElementId || 'videoElement',
            canvasElementId: config.canvasElementId || 'canvasElement',
            statusElementId: config.statusElementId || 'status-message',
            progressElementId: config.progressElementId || 'progress-bar',
            ...config
        };
        
        this.socket = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        
        // State management
        this.currentState = 'init';
        this.sessionId = null;
        this.isProcessing = false;
        this.frameInterval = null;
        
        // Performance optimization
        this.frameSkipCounter = 0;
        this.frameSkipRate = 3; // Send every 3rd frame
        
        this.init();
    }
    
    async init() {
        try {
            console.log('Initializing Step 5 Attendance Flow...');
            
            // Get DOM elements
            this.video = document.getElementById(this.config.videoElementId);
            this.canvas = document.getElementById(this.config.canvasElementId);
            this.ctx = this.canvas.getContext('2d');
            
            // Initialize socket connection
            this.initSocket();
            
            // Initialize camera
            await this.initCamera();
            
            console.log('Step 5 Attendance Flow initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize Step 5 system:', error);
            this.updateStatus('Initialization failed: ' + error.message, 'error');
        }
    }
    
    initSocket() {
        this.socket = io(this.config.socketUrl);
        
        // Socket event handlers
        this.socket.on('connect', () => {
            console.log('Connected to Step 5 server');
            this.updateStatus('Connected to server', 'info');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from Step 5 server');
            this.updateStatus('Disconnected from server', 'warning');
        });
        
        // Step 5 specific events
        this.socket.on('session_started', (data) => {
            this.handleSessionStarted(data);
        });
        
        this.socket.on('state_update', (data) => {
            this.handleStateUpdate(data);
        });
        
        this.socket.on('antispoofing_progress', (data) => {
            this.handleAntispoofingProgress(data);
        });
        
        this.socket.on('recognition_started', (data) => {
            this.handleRecognitionStarted(data);
        });
        
        this.socket.on('attendance_success', (data) => {
            this.handleAttendanceSuccess(data);
        });
        
        this.socket.on('attendance_failed', (data) => {
            this.handleAttendanceFailed(data);
        });
        
        this.socket.on('session_timeout', (data) => {
            this.handleSessionTimeout(data);
        });
        
        this.socket.on('error', (data) => {
            this.handleError(data);
        });
    }
    
    async initCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.video.onloadedmetadata = () => {
                this.video.play();
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            };
            
        } catch (error) {
            throw new Error('Camera access failed: ' + error.message);
        }
    }
    
    startAttendance() {
        if (this.isProcessing) {
            console.log('Attendance already in progress');
            return;
        }
        
        console.log('Starting Step 5 attendance process...');
        this.isProcessing = true;
        
        // Request session start from server
        this.socket.emit('start_step5_attendance', {
            timestamp: Date.now()
        });
        
        // Start sending frames
        this.startFrameCapture();
    }
    
    startFrameCapture() {
        // Stop any existing interval
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
        }
        
        // Send frames at 10 FPS (with skipping for performance)
        this.frameInterval = setInterval(() => {
            if (this.isProcessing && this.video.readyState === 4) {
                this.captureAndSendFrame();
            }
        }, 100); // 10 FPS
    }
    
    captureAndSendFrame() {
        // Performance optimization: frame skipping
        this.frameSkipCounter++;
        if (this.frameSkipCounter < this.frameSkipRate) {
            return;
        }
        this.frameSkipCounter = 0;
        
        try {
            // Draw video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert to blob and send
            this.canvas.toBlob((blob) => {
                if (blob && this.sessionId) {
                    const formData = new FormData();
                    formData.append('image', blob);
                    formData.append('session_id', this.sessionId);
                    formData.append('timestamp', Date.now());
                    
                    // Send via socket.io
                    this.socket.emit('process_frame_step5', {
                        session_id: this.sessionId,
                        timestamp: Date.now()
                    });
                }
            }, 'image/jpeg', 0.8);
            
        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }
    
    stopAttendance() {
        console.log('Stopping Step 5 attendance process...');
        
        this.isProcessing = false;
        
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        if (this.sessionId) {
            this.socket.emit('stop_session', {
                session_id: this.sessionId
            });
        }
        
        this.resetSession();
    }
    
    resetSession() {
        this.currentState = 'init';
        this.sessionId = null;
        this.isProcessing = false;
        this.frameSkipCounter = 0;
        
        this.updateStatus('Ready to start attendance', 'info');
        this.updateProgress(0);
        this.hideStateIndicators();
    }
    
    // Event Handlers
    
    handleSessionStarted(data) {
        console.log('Session started:', data);
        this.sessionId = data.session_id;
        this.currentState = 'anti_spoofing';
        
        this.updateStatus('Session started - Verifying real person...', 'info');
        this.showStateIndicator('antispoofing', true);
    }
    
    handleStateUpdate(data) {
        console.log('State update:', data);
        this.currentState = data.state;
        
        switch (data.state) {
            case 'anti_spoofing':
                this.showStateIndicator('antispoofing', true);
                this.showStateIndicator('recognition', false);
                break;
                
            case 'recognizing':
                this.showStateIndicator('antispoofing', false);
                this.showStateIndicator('recognition', true);
                break;
                
            case 'success':
            case 'failed':
                this.showStateIndicator('antispoofing', false);
                this.showStateIndicator('recognition', false);
                break;
        }
        
        this.updateStatus(data.message, data.status);
    }
    
    handleAntispoofingProgress(data) {
        console.log('Anti-spoofing progress:', data);
        
        // Update progress bar
        const progress = data.progress || 0;
        this.updateProgress(progress * 100);
        
        // Update status message
        this.updateStatus(data.message, data.status);
        
        // Show challenge info if available
        if (data.challenge_info) {
            this.showChallengeInfo(data.challenge_info);
        }
        
        // Add visual indicators based on confidence
        if (data.confidence) {
            this.updateConfidenceIndicator('antispoofing', data.confidence);
        }
    }
    
    handleRecognitionStarted(data) {
        console.log('Recognition started:', data);
        
        this.currentState = 'recognizing';
        this.showStateIndicator('antispoofing', false);
        this.showStateIndicator('recognition', true);
        
        this.updateStatus('Face verified! Recognizing...', 'success');
        this.updateProgress(100); // Anti-spoofing complete
        
        // Show antispoofing confidence
        if (data.antispoofing_confidence) {
            this.updateConfidenceIndicator('antispoofing', data.antispoofing_confidence);
        }
    }
    
    handleAttendanceSuccess(data) {
        console.log('Attendance success:', data);
        
        this.currentState = 'success';
        this.isProcessing = false;
        
        // Stop frame capture
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
        }
        
        // Update UI
        this.updateStatus(data.message, 'success');
        this.showStateIndicator('recognition', false);
        this.showSuccessIndicator(true);
        
        // Show user info
        if (data.user_info) {
            this.displayUserInfo(data.user_info);
        }
        
        // Show confidence scores
        if (data.recognition_confidence) {
            this.updateConfidenceIndicator('recognition', data.recognition_confidence);
        }
        if (data.antispoofing_confidence) {
            this.updateConfidenceIndicator('antispoofing', data.antispoofing_confidence);
        }
        
        // Show performance metrics
        if (data.total_time) {
            this.displayPerformanceMetrics(data);
        }
        
        // Auto-reset after 5 seconds
        setTimeout(() => {
            this.resetSession();
        }, 5000);
    }
    
    handleAttendanceFailed(data) {
        console.log('Attendance failed:', data);
        
        this.currentState = 'failed';
        this.isProcessing = false;
        
        // Stop frame capture
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
        }
        
        // Update UI
        this.updateStatus(data.message, 'error');
        this.showStateIndicator('antispoofing', false);
        this.showStateIndicator('recognition', false);
        this.showErrorIndicator(true);
        
        // Auto-reset after 3 seconds
        setTimeout(() => {
            this.resetSession();
        }, 3000);
    }
    
    handleSessionTimeout(data) {
        console.log('Session timeout:', data);
        
        this.updateStatus(data.message, 'warning');
        this.stopAttendance();
    }
    
    handleError(data) {
        console.error('Step 5 error:', data);
        
        this.updateStatus('Error: ' + data.message, 'error');
        this.stopAttendance();
    }
    
    // UI Update Methods
    
    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById(this.config.statusElementId);
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status-message ${type}`;
        }
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    updateProgress(percentage) {
        const progressElement = document.getElementById(this.config.progressElementId);
        if (progressElement) {
            progressElement.style.width = `${percentage}%`;
            progressElement.setAttribute('aria-valuenow', percentage);
        }
    }
    
    showStateIndicator(phase, active) {
        const indicatorId = `${phase}-indicator`;
        const indicator = document.getElementById(indicatorId);
        
        if (indicator) {
            if (active) {
                indicator.classList.add('active');
                indicator.classList.remove('completed');
            } else {
                indicator.classList.remove('active');
                if (phase === 'antispoofing' && this.currentState === 'recognizing') {
                    indicator.classList.add('completed');
                }
            }
        }
    }
    
    hideStateIndicators() {
        ['antispoofing', 'recognition'].forEach(phase => {
            this.showStateIndicator(phase, false);
            const indicator = document.getElementById(`${phase}-indicator`);
            if (indicator) {
                indicator.classList.remove('completed');
            }
        });
        
        this.showSuccessIndicator(false);
        this.showErrorIndicator(false);
    }
    
    showSuccessIndicator(show) {
        const indicator = document.getElementById('success-indicator');
        if (indicator) {
            indicator.style.display = show ? 'block' : 'none';
        }
    }
    
    showErrorIndicator(show) {
        const indicator = document.getElementById('error-indicator');
        if (indicator) {
            indicator.style.display = show ? 'block' : 'none';
        }
    }
    
    updateConfidenceIndicator(type, confidence) {
        const indicatorId = `${type}-confidence`;
        const indicator = document.getElementById(indicatorId);
        
        if (indicator) {
            const percentage = Math.round(confidence * 100);
            indicator.textContent = `${percentage}%`;
            
            // Color based on confidence level
            indicator.className = 'confidence-indicator';
            if (confidence >= 0.85) {
                indicator.classList.add('high');
            } else if (confidence >= 0.70) {
                indicator.classList.add('medium');
            } else {
                indicator.classList.add('low');
            }
        }
    }
    
    showChallengeInfo(challengeInfo) {
        const challengeElement = document.getElementById('challenge-instruction');
        if (challengeElement && challengeInfo.instruction) {
            challengeElement.textContent = challengeInfo.instruction;
            challengeElement.style.display = 'block';
        }
    }
    
    displayUserInfo(userInfo) {
        const userInfoElement = document.getElementById('user-info');
        if (userInfoElement) {
            userInfoElement.innerHTML = `
                <h4>Welcome, ${userInfo.name}!</h4>
                <p>ID: ${userInfo.user_id}</p>
                <p>Department: ${userInfo.department || 'N/A'}</p>
                <p>Role: ${userInfo.role || 'Employee'}</p>
            `;
            userInfoElement.style.display = 'block';
        }
    }
    
    displayPerformanceMetrics(data) {
        const metricsElement = document.getElementById('performance-metrics');
        if (metricsElement) {
            metricsElement.innerHTML = `
                <small>
                    Total Time: ${data.total_time?.toFixed(2)}s |
                    Processing: ${data.processing_time?.toFixed(2)}s |
                    Anti-spoofing: ${data.antispoofing_confidence ? (data.antispoofing_confidence * 100).toFixed(1) + '%' : 'N/A'} |
                    Recognition: ${data.recognition_confidence ? (data.recognition_confidence * 100).toFixed(1) + '%' : 'N/A'}
                </small>
            `;
            metricsElement.style.display = 'block';
        }
    }
    
    // Public API
    
    getStatus() {
        return {
            currentState: this.currentState,
            sessionId: this.sessionId,
            isProcessing: this.isProcessing
        };
    }
    
    destroy() {
        this.stopAttendance();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Step5AttendanceFlow;
} else {
    window.Step5AttendanceFlow = Step5AttendanceFlow;
}