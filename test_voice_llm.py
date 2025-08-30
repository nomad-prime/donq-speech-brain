#!/usr/bin/env python3
"""
Unit tests for Voice-to-LLM Application
"""

import unittest
import tempfile
import os
import wave
import numpy as np
import httpx
from unittest.mock import Mock, patch, MagicMock
from voice_llm_app import (
    AudioConfig, APIConfig, AudioCapture, 
    SilenceDetector, WhisperProcessor, TogetherClient
)


class TestAudioConfig(unittest.TestCase):
    """Test AudioConfig dataclass"""
    
    def test_default_values(self):
        config = AudioConfig()
        self.assertEqual(config.CHUNK_SIZE, 1024)
        self.assertEqual(config.SAMPLE_RATE, 16000)
        self.assertEqual(config.CHANNELS, 1)
        self.assertEqual(config.SILENCE_THRESHOLD, 500)
        self.assertEqual(config.SILENCE_DURATION, 2.5)
        self.assertEqual(config.MIN_AUDIO_LENGTH, 1.0)


class TestSilenceDetector(unittest.TestCase):
    """Test SilenceDetector class"""
    
    def setUp(self):
        self.config = AudioConfig()
        self.detector = SilenceDetector(self.config)
    
    def test_calculate_rms_silent(self):
        # Create silent audio (all zeros)
        silent_chunk = np.zeros(self.config.CHUNK_SIZE, dtype=np.int16).tobytes()
        rms = self.detector.calculate_rms(silent_chunk)
        self.assertEqual(rms, 0)
    
    def test_calculate_rms_loud(self):
        # Create loud audio
        loud_chunk = np.full(self.config.CHUNK_SIZE, 1000, dtype=np.int16).tobytes()
        rms = self.detector.calculate_rms(loud_chunk)
        self.assertGreater(rms, self.config.SILENCE_THRESHOLD)
    
    def test_detect_speech_start(self):
        # Create audio above threshold
        loud_chunk = np.full(self.config.CHUNK_SIZE, 2000, dtype=np.int16).tobytes()
        is_silent, trigger = self.detector.process_chunk(loud_chunk)
        
        self.assertFalse(is_silent)
        self.assertFalse(trigger)
        self.assertTrue(self.detector.is_speaking)
    
    def test_detect_silence_after_speech(self):
        # First, simulate speech
        loud_chunk = np.full(self.config.CHUNK_SIZE, 2000, dtype=np.int16).tobytes()
        self.detector.process_chunk(loud_chunk)
        
        # Then simulate silence
        silent_chunk = np.zeros(self.config.CHUNK_SIZE, dtype=np.int16).tobytes()
        is_silent, trigger = self.detector.process_chunk(silent_chunk)
        
        self.assertTrue(is_silent)
        self.assertFalse(trigger)  # Not enough duration yet
        self.assertIsNotNone(self.detector.silence_start)


class TestAudioCapture(unittest.TestCase):
    """Test AudioCapture class"""
    
    def setUp(self):
        self.config = AudioConfig()
        
    @patch('pyaudio.PyAudio')
    def test_initialization(self, mock_pyaudio):
        capture = AudioCapture(self.config)
        self.assertIsNotNone(capture.audio)
        self.assertEqual(len(capture.audio_buffer), 0)
        self.assertFalse(capture.is_recording)
    
    @patch('pyaudio.PyAudio')
    def test_buffer_operations(self, mock_pyaudio):
        capture = AudioCapture(self.config)
        
        # Test adding to buffer
        test_chunk = b'test_audio_data'
        capture.add_to_buffer(test_chunk)
        self.assertEqual(len(capture.audio_buffer), 1)
        
        # Test getting buffer copy
        buffer_copy = capture.get_buffer_copy()
        self.assertEqual(buffer_copy[0], test_chunk)
        
        # Test clearing buffer
        capture.clear_buffer()
        self.assertEqual(len(capture.audio_buffer), 0)


class TestWhisperProcessor(unittest.TestCase):
    """Test WhisperProcessor class"""
    
    def setUp(self):
        self.processor = WhisperProcessor()
        self.config = AudioConfig()
    
    def test_initialization(self):
        self.assertIsNone(self.processor.model)
        self.assertFalse(self.processor.model_loaded)
    
    def test_load_model_success(self):
        # Skip if whisper not installed
        try:
            import whisper
            with patch('whisper.load_model') as mock_load:
                mock_load.return_value = Mock()
                result = self.processor.load_model()
                self.assertTrue(result)
                self.assertTrue(self.processor.model_loaded)
        except ImportError:
            self.skipTest("Whisper not installed")
    
    def test_load_model_failure(self):
        # Skip if whisper not installed
        try:
            import whisper
            with patch('whisper.load_model') as mock_load:
                mock_load.side_effect = Exception("Model load failed")
                result = self.processor.load_model()
                self.assertFalse(result)
                self.assertFalse(self.processor.model_loaded)
        except ImportError:
            self.skipTest("Whisper not installed")
    
    def test_transcribe_without_model(self):
        audio_buffer = [b'test'] * 100
        result = self.processor.transcribe(audio_buffer, self.config)
        self.assertIsNone(result)
    
    def test_audio_too_short(self):
        self.processor.model_loaded = True
        self.processor.model = Mock()
        
        # Create very short audio buffer (less than MIN_AUDIO_LENGTH)
        audio_buffer = [b'x' * self.config.CHUNK_SIZE]
        result = self.processor.transcribe(audio_buffer, self.config)
        self.assertIsNone(result)


class TestTogetherClient(unittest.TestCase):
    """Test TogetherClient class"""
    
    def setUp(self):
        self.config = APIConfig()
        self.config.TOGETHER_API_KEY = "test_key"
        
    def test_initialization(self):
        client = TogetherClient(self.config)
        self.assertIsNotNone(client.client)
        self.assertEqual(client.config.TOGETHER_API_KEY, "test_key")
    
    def test_no_api_key(self):
        self.config.TOGETHER_API_KEY = ""
        client = TogetherClient(self.config)
        result = client.send_message("test")
        self.assertIsNone(result)
    
    @patch('httpx.Client.post')
    def test_send_message_success(self, mock_post):
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response from LLM"
                }
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = TogetherClient(self.config)
        result = client.send_message("Test input")
        
        self.assertEqual(result, "Test response from LLM")
        mock_post.assert_called_once()
    
    @patch('httpx.Client.post')
    def test_send_message_http_error(self, mock_post):
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", 
            request=Mock(), 
            response=mock_response
        )
        mock_post.return_value = mock_response
        
        client = TogetherClient(self.config)
        result = client.send_message("Test input")
        
        self.assertIsNone(result)


class TestIntegration(unittest.TestCase):
    """Integration tests for component interaction"""
    
    def test_audio_config_integration(self):
        # Test that AudioConfig is properly used by other components
        config = AudioConfig()
        detector = SilenceDetector(config)
        
        # Verify detector uses config values
        silent_chunk = np.zeros(config.CHUNK_SIZE, dtype=np.int16).tobytes()
        rms = detector.calculate_rms(silent_chunk)
        self.assertLess(rms, config.SILENCE_THRESHOLD)
    
    def test_audio_buffer_to_wav_format(self):
        # Test that audio buffer can be properly formatted for WAV file
        config = AudioConfig()
        
        # Create test audio data
        test_samples = 1000
        audio_data = np.random.randint(-1000, 1000, test_samples, dtype=np.int16)
        audio_bytes = audio_data.tobytes()
        
        # Write to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(config.CHANNELS)
                wav_file.setsampwidth(config.BYTES_PER_SAMPLE)
                wav_file.setframerate(config.SAMPLE_RATE)
                wav_file.writeframes(audio_bytes)
            
            # Verify file was created and can be read
            with wave.open(temp_path, 'rb') as wav_file:
                self.assertEqual(wav_file.getnchannels(), config.CHANNELS)
                self.assertEqual(wav_file.getsampwidth(), config.BYTES_PER_SAMPLE)
                self.assertEqual(wav_file.getframerate(), config.SAMPLE_RATE)
                
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()