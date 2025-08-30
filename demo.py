#!/usr/bin/env python3
"""
Demo script to test individual components of the Voice-to-LLM application
"""

import numpy as np
import time
from voice_llm_app import AudioConfig, SilenceDetector

def demo_silence_detection():
    """Demonstrate silence detection with simulated audio"""
    print("=" * 50)
    print("SILENCE DETECTION DEMO")
    print("=" * 50)
    
    config = AudioConfig()
    detector = SilenceDetector(config)
    
    # Simulate speaking (loud audio)
    print("\n1. Simulating speech...")
    for i in range(5):
        loud_chunk = np.random.randint(800, 2000, config.CHUNK_SIZE, dtype=np.int16).tobytes()
        is_silent, trigger = detector.process_chunk(loud_chunk)
        print(f"   Frame {i+1}: Silent={is_silent}, Trigger={trigger}")
        time.sleep(0.1)
    
    # Simulate silence
    print("\n2. Simulating silence...")
    for i in range(30):  # About 3 seconds of silence
        silent_chunk = np.random.randint(-100, 100, config.CHUNK_SIZE, dtype=np.int16).tobytes()
        is_silent, trigger = detector.process_chunk(silent_chunk)
        if trigger:
            print(f"   Frame {i+1}: SILENCE DETECTED - Processing triggered!")
            break
        elif i % 5 == 0:
            print(f"   Frame {i+1}: Silent={is_silent}, Waiting...")
        time.sleep(0.1)
    
    print("\n✅ Silence detection demo complete!")

def demo_rms_calculation():
    """Demonstrate RMS calculation for different audio levels"""
    print("\n" + "=" * 50)
    print("RMS CALCULATION DEMO")
    print("=" * 50)
    
    config = AudioConfig()
    detector = SilenceDetector(config)
    
    test_cases = [
        ("Silent", np.zeros(config.CHUNK_SIZE, dtype=np.int16)),
        ("Very Quiet", np.random.randint(-50, 50, config.CHUNK_SIZE, dtype=np.int16)),
        ("Quiet", np.random.randint(-200, 200, config.CHUNK_SIZE, dtype=np.int16)),
        ("Normal Speech", np.random.randint(-1000, 1000, config.CHUNK_SIZE, dtype=np.int16)),
        ("Loud Speech", np.random.randint(-3000, 3000, config.CHUNK_SIZE, dtype=np.int16)),
    ]
    
    print(f"\nThreshold: {config.SILENCE_THRESHOLD}")
    print("-" * 30)
    
    for name, audio_array in test_cases:
        rms = detector.calculate_rms(audio_array.tobytes())
        status = "🔇 SILENT" if rms < config.SILENCE_THRESHOLD else "🗣️ SPEECH"
        print(f"{name:15} | RMS: {rms:7.1f} | {status}")
    
    print("\n✅ RMS calculation demo complete!")

def main():
    """Run all demos"""
    print("\n🎤 VOICE-TO-LLM COMPONENT DEMOS")
    print("This demonstrates the core components without requiring API keys")
    print("⚠️ Note: This demo requires Python 3.11\n")
    
    demo_rms_calculation()
    demo_silence_detection()
    
    print("\n" + "=" * 50)
    print("📌 To run the full application:")
    print("1. Set TOGETHER_API_KEY environment variable")
    print("2. Run: uv run python voice_llm_app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()