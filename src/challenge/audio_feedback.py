"""
Audio Feedback System for Challenge-Response
Implements success/failure sounds and voice instructions
"""

import os
import threading
import queue
import time
from typing import Dict, List, Optional
from enum import Enum
import json

try:
    # Try to import pygame for audio playback
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Audio feedback disabled.")

try:
    # Try to import pyttsx3 for text-to-speech
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Voice instructions disabled.")


class AudioType(Enum):
    """Types of audio feedback"""
    SUCCESS = "success"
    FAILURE = "failure" 
    WARNING = "warning"
    INSTRUCTION = "instruction"
    PROGRESS = "progress"
    COUNTDOWN = "countdown"


class AudioFeedbackSystem:
    """
    Handles audio feedback for challenge-response system
    Supports both sound effects and text-to-speech instructions
    """
    
    def __init__(self, audio_enabled: bool = True, voice_enabled: bool = True):
        self.audio_enabled = audio_enabled and PYGAME_AVAILABLE
        self.voice_enabled = voice_enabled and TTS_AVAILABLE
        
        # Audio queue for thread-safe playback
        self.audio_queue = queue.Queue()
        self.voice_queue = queue.Queue()
        
        # Initialize TTS engine
        self.tts_engine = None
        if self.voice_enabled:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speaking rate
                self.tts_engine.setProperty('volume', 0.8)  # Volume level
                
                # Get available voices and set to English if available
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                        
            except Exception as e:
                print(f"TTS initialization failed: {e}")
                self.voice_enabled = False
        
        # Sound effects storage
        self.sounds = {}
        self._initialize_sounds()
        
        # Start background threads
        self.running = True
        if self.audio_enabled:
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
        
        if self.voice_enabled:
            self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
            self.voice_thread.start()
    
    def _initialize_sounds(self):
        """Initialize or generate sound effects"""
        if not self.audio_enabled:
            return
        
        # Generate simple sound effects using pygame
        try:
            # Success sound (ascending notes)
            self.sounds[AudioType.SUCCESS] = self._generate_success_sound()
            
            # Failure sound (descending notes)
            self.sounds[AudioType.FAILURE] = self._generate_failure_sound()
            
            # Warning sound (single beep)
            self.sounds[AudioType.WARNING] = self._generate_warning_sound()
            
            # Progress sound (short beep)
            self.sounds[AudioType.PROGRESS] = self._generate_progress_sound()
            
            # Countdown sound (tick)
            self.sounds[AudioType.COUNTDOWN] = self._generate_countdown_sound()
            
        except Exception as e:
            print(f"Sound generation failed: {e}")
            self.audio_enabled = False
    
    def _generate_success_sound(self):
        """Generate success sound (ascending chord)"""
        duration = 0.5
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Generate ascending notes
        freqs = [523, 659, 784]  # C, E, G notes
        wave = np.zeros(samples)
        
        for freq in freqs:
            t = np.linspace(0, duration, samples)
            note = np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)  # Decay envelope
            wave += note / len(freqs)
        
        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _generate_failure_sound(self):
        """Generate failure sound (descending notes)"""
        duration = 0.8
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Generate descending notes
        freqs = [400, 350, 300]  # Descending tones
        wave = np.zeros(samples)
        
        segment_length = samples // len(freqs)
        for i, freq in enumerate(freqs):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, samples)
            segment_samples = end_idx - start_idx
            
            t = np.linspace(0, segment_samples/sample_rate, segment_samples)
            note = np.sin(2 * np.pi * freq * t) * np.exp(-t * 2)
            wave[start_idx:end_idx] = note
        
        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _generate_warning_sound(self):
        """Generate warning sound (beep)"""
        duration = 0.3
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        freq = 800  # Warning frequency
        t = np.linspace(0, duration, samples)
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 5)
        
        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _generate_progress_sound(self):
        """Generate progress sound (short beep)"""
        duration = 0.15
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        freq = 600  # Progress frequency
        t = np.linspace(0, duration, samples)
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 8)
        
        # Convert to 16-bit integers
        wave = (wave * 16383).astype(np.int16)  # Quieter for frequent use
        
        # Create stereo sound
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _generate_countdown_sound(self):
        """Generate countdown sound (tick)"""
        duration = 0.1
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        freq = 1000  # Tick frequency
        t = np.linspace(0, duration, samples)
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15)
        
        # Convert to 16-bit integers
        wave = (wave * 16383).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)
    
    def _audio_worker(self):
        """Background thread for audio playback"""
        while self.running:
            try:
                audio_type = self.audio_queue.get(timeout=1.0)
                if audio_type in self.sounds:
                    self.sounds[audio_type].play()
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def _voice_worker(self):
        """Background thread for voice synthesis"""
        while self.running:
            try:
                text = self.voice_queue.get(timeout=1.0)
                if self.tts_engine:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                self.voice_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice synthesis error: {e}")
    
    def play_sound(self, audio_type: AudioType):
        """Play sound effect"""
        if self.audio_enabled:
            try:
                self.audio_queue.put_nowait(audio_type)
            except queue.Full:
                pass  # Skip if queue is full
    
    def speak(self, text: str):
        """Speak text using TTS"""
        if self.voice_enabled:
            try:
                self.voice_queue.put_nowait(text)
            except queue.Full:
                pass  # Skip if queue is full
    
    def play_challenge_start(self, challenge_description: str):
        """Play audio for challenge start"""
        self.speak(f"New challenge: {challenge_description}")
        self.play_sound(AudioType.INSTRUCTION)
    
    def play_challenge_success(self, challenge_type: str):
        """Play audio for challenge success"""
        self.speak(f"{challenge_type} challenge completed successfully!")
        self.play_sound(AudioType.SUCCESS)
    
    def play_challenge_failure(self, challenge_type: str, reason: str = ""):
        """Play audio for challenge failure"""
        message = f"{challenge_type} challenge failed."
        if reason:
            message += f" {reason}"
        self.speak(message)
        self.play_sound(AudioType.FAILURE)
    
    def play_progress_update(self, progress_text: str):
        """Play audio for progress update"""
        self.speak(progress_text)
        self.play_sound(AudioType.PROGRESS)
    
    def play_warning(self, warning_text: str):
        """Play warning audio"""
        self.speak(warning_text)
        self.play_sound(AudioType.WARNING)
    
    def play_countdown(self, seconds_remaining: int):
        """Play countdown audio"""
        if seconds_remaining <= 5:
            self.speak(f"{seconds_remaining}")
            self.play_sound(AudioType.COUNTDOWN)
    
    def set_audio_enabled(self, enabled: bool):
        """Enable/disable sound effects"""
        self.audio_enabled = enabled and PYGAME_AVAILABLE
    
    def set_voice_enabled(self, enabled: bool):
        """Enable/disable voice instructions"""
        self.voice_enabled = enabled and TTS_AVAILABLE
    
    def shutdown(self):
        """Shutdown audio system"""
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)
        if hasattr(self, 'voice_thread'):
            self.voice_thread.join(timeout=1.0)


# Enhanced Challenge Instructions
class ChallengeInstructions:
    """
    Provides detailed instructions for each challenge type
    """
    
    INSTRUCTIONS = {
        'blink': {
            'start': "Please blink your eyes naturally {count} times",
            'progress': "Blink detected {current} of {total}",
            'success': "Blink challenge completed",
            'failure': "Blink challenge timeout. Please try again.",
            'guidance': [
                "Blink slowly and naturally",
                "Make sure your eyes close completely",
                "Avoid rapid or fake blinking"
            ]
        },
        'smile': {
            'start': "Please smile naturally {count} times",
            'progress': "Smile detected {current} of {total}",
            'success': "Smile challenge completed",
            'failure': "Smile challenge timeout. Please try again.",
            'guidance': [
                "Smile naturally with your mouth",
                "Hold each smile for at least half a second",
                "Return to neutral expression between smiles"
            ]
        },
        'head_movement': {
            'start': "Please turn your head {directions}",
            'progress': "Completed {current} of {total} movements",
            'success': "Head movement challenge completed",
            'failure': "Head movement challenge timeout. Please try again.",
            'guidance': [
                "Turn your head slowly and deliberately",
                "Hold each position for at least 1 second",
                "Return to center between movements"
            ]
        },
        'move_closer': {
            'start': "Please move your face closer to the camera",
            'progress': "Distance progress: {progress}%",
            'success': "Distance challenge completed",
            'failure': "Distance challenge timeout. Please try again.",
            'guidance': [
                "Move slowly towards the camera",
                "Stop when instructed to hold position",
                "Keep your face visible in the frame"
            ]
        },
        'move_farther': {
            'start': "Please move your face farther from the camera",
            'progress': "Distance progress: {progress}%",
            'success': "Distance challenge completed",
            'failure': "Distance challenge timeout. Please try again.",
            'guidance': [
                "Move slowly away from the camera",
                "Stop when instructed to hold position",
                "Keep your face visible in the frame"
            ]
        }
    }
    
    @classmethod
    def get_instruction(cls, challenge_type: str, instruction_type: str, **kwargs):
        """Get specific instruction text"""
        instructions = cls.INSTRUCTIONS.get(challenge_type, {})
        template = instructions.get(instruction_type, "")
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    @classmethod
    def get_guidance(cls, challenge_type: str) -> List[str]:
        """Get guidance list for challenge type"""
        instructions = cls.INSTRUCTIONS.get(challenge_type, {})
        return instructions.get('guidance', [])


# Import numpy for sound generation
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available. Sound generation disabled.")
    PYGAME_AVAILABLE = False
