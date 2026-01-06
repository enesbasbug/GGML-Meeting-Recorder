#!/usr/bin/env python3
"""
Meeting Recorder Pro - GGML-Powered Menu Bar App with Model Selection
======================================================================

Features:
- Multiple LLM models to choose from (DeepSeek, Qwen, Mistral, Llama)
- Multiple Whisper models (Base for speed, Large for quality)
- Automatic transcription, summarization, and to-do extraction
- All processing runs locally using GGML on Apple Silicon

Usage: python3 MeetingRecorderPro.py
"""

import rumps
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import threading
import subprocess
import os
from datetime import datetime
from pathlib import Path
from Foundation import NSObject
from PyObjCTools import AppHelper
import objc


# Model configurations with their chat templates
# Sorted by quality (best first)
LLAMA_MODELS = {
    "⭐ Llama 3.3 70B (Best Quality)": {
        "file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "template": "llama3",
        "size": "40GB",
        "description": "Latest Llama, best for summarization"
    },
    "⭐ DeepSeek R1 32B (Best Reasoning)": {
        "file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "template": "chatml",  # Qwen-based uses chatml
        "size": "18GB",
        "description": "Excellent reasoning, 2x faster than 70B"
    },
    "DeepSeek R1 8B (Fast Reasoning)": {
        "file": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "template": "llama3",
        "size": "4.6GB",
        "description": "Quick reasoning tasks"
    },
    "Qwen 2.5 7B (Summarization)": {
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "template": "chatml",
        "size": "4.4GB",
        "description": "Great for summarization"
    },
    "Mistral 7B (Balanced)": {
        "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "template": "mistral",
        "size": "4.1GB",
        "description": "Fast & capable"
    },
    "Llama 3.2 3B (Fastest)": {
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "template": "llama3",
        "size": "1.9GB",
        "description": "Quick responses, testing"
    }
}

WHISPER_MODELS = {
    "Whisper Base (Fast)": {
        "file": "ggml-base.en.bin",
        "size": "141MB",
        "description": "~150ms for 10s audio"
    },
    "Whisper Large V3 (Best)": {
        "file": "ggml-large-v3.bin",
        "size": "2.9GB",
        "description": "Best accuracy"
    }
}


def get_prompt_template(template_type, system_prompt, user_message):
    """Generate the appropriate prompt format for each model"""
    if template_type == "llama3":
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    elif template_type == "chatml":
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    elif template_type == "mistral":
        return f"""[INST] {system_prompt}

{user_message} [/INST]"""
    
    else:
        # Fallback
        return f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"


class MeetingRecorderProApp(rumps.App):
    def __init__(self):
        super(MeetingRecorderProApp, self).__init__(
            "Meeting Recorder Pro",
            title="🎙️",
            quit_button="Quit"
        )
        
        # Configuration paths
        self.base_dir = Path(__file__).parent.absolute()
        self.recordings_dir = self.base_dir / "recordings"
        self.models_dir = self.base_dir / "models"
        self.whisper_cli = self.base_dir / "whisper.cpp" / "build" / "bin" / "whisper-cli"
        self.whisper_models_dir = self.base_dir / "whisper.cpp" / "models"
        self.llama_cli = self.base_dir / "llama.cpp" / "build" / "bin" / "llama-cli"
        
        # Current model selection (default to best available)
        self.current_llm = "⭐ Llama 3.3 70B (Best Quality)"
        self.current_whisper = "Whisper Large V3 (Best)"
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.stream = None
        self.current_session_dir = None
        
        # Create directories
        self.recordings_dir.mkdir(exist_ok=True)
        
        # Build menu
        self.build_menu()
    
    def build_menu(self):
        """Build the menu with model selectors"""
        # Recording controls
        self.record_button = rumps.MenuItem("▶️ Start Recording", callback=self.toggle_recording)
        self.status_item = rumps.MenuItem("Status: Ready")
        
        # LLM Model selector
        llm_menu = rumps.MenuItem("🧠 LLM Model")
        for model_name, model_info in LLAMA_MODELS.items():
            model_path = self.models_dir / model_info["file"]
            available = "✓" if model_path.exists() else "✗"
            item = rumps.MenuItem(
                f"{available} {model_name} ({model_info['size']})",
                callback=lambda sender, m=model_name: self.select_llm(m)
            )
            if model_name == self.current_llm:
                item.state = 1  # Checkmark
            llm_menu.add(item)
        
        # Whisper Model selector
        whisper_menu = rumps.MenuItem("🎤 Whisper Model")
        for model_name, model_info in WHISPER_MODELS.items():
            model_path = self.whisper_models_dir / model_info["file"]
            available = "✓" if model_path.exists() else "✗"
            item = rumps.MenuItem(
                f"{available} {model_name} ({model_info['size']})",
                callback=lambda sender, m=model_name: self.select_whisper(m)
            )
            if model_name == self.current_whisper:
                item.state = 1
            whisper_menu.add(item)
        
        # Other menu items
        self.open_folder = rumps.MenuItem("📁 Open Recordings", callback=self.open_recordings)
        self.about_item = rumps.MenuItem("ℹ️ About", callback=self.show_about)
        
        self.menu = [
            self.record_button,
            None,
            self.status_item,
            None,
            llm_menu,
            whisper_menu,
            None,
            self.open_folder,
            self.about_item
        ]
    
    def select_llm(self, model_name):
        """Select a different LLM model"""
        self.current_llm = model_name
        # Update checkmarks
        for item in self.menu["🧠 LLM Model"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 1 if model_name in item.title else 0
        rumps.notification(
            title="Model Changed",
            subtitle=f"LLM: {model_name}",
            message=LLAMA_MODELS[model_name]["description"],
            sound=False
        )
    
    def select_whisper(self, model_name):
        """Select a different Whisper model"""
        self.current_whisper = model_name
        # Update checkmarks
        for item in self.menu["🎤 Whisper Model"].values():
            if isinstance(item, rumps.MenuItem):
                item.state = 1 if model_name in item.title else 0
        rumps.notification(
            title="Model Changed",
            subtitle=f"Whisper: {model_name}",
            message=WHISPER_MODELS[model_name]["description"],
            sound=False
        )
    
    def show_about(self, sender):
        """Show about dialog"""
        rumps.alert(
            title="Meeting Recorder Pro",
            message="""Powered by GGML on Apple Silicon

🎤 Speech-to-Text: whisper.cpp
🧠 Summarization: llama.cpp

Models available:
• DeepSeek R1 - State-of-the-art reasoning
• Qwen 2.5 - Excellent summarization  
• Mistral 7B - Fast & capable
• Llama 3.2 - Quick responses

All processing runs 100% locally.
Zero cloud, zero API costs!

Built with ❤️ using GGML"""
        )
    
    def toggle_recording(self, sender):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.audio_data = []
        self.title = "🔴"
        self.record_button.title = "⏹️ Stop Recording"
        self.status_item.title = "Status: 🔴 Recording..."
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_session_dir = self.recordings_dir / f"meeting_{timestamp}"
        self.current_session_dir.mkdir(exist_ok=True)
        
        # Save model info
        info_file = self.current_session_dir / "model_info.txt"
        info_file.write_text(f"Whisper: {self.current_whisper}\nLLM: {self.current_llm}\n")
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_data.append(indata.copy())
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback
            )
            self.stream.start()
            rumps.notification(
                title="Recording Started",
                subtitle=f"Using {self.current_whisper.split('(')[0].strip()}",
                message="Click the menu bar icon to stop",
                sound=False
            )
        except Exception as e:
            rumps.alert(f"Error: {e}")
            self.reset_ui()
    
    def stop_recording(self):
        """Stop recording and process"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.is_recording = False
        self.title = "⏳"
        self.record_button.title = "⏳ Processing..."
        self.status_item.title = "Status: ⏳ Processing..."
        
        threading.Thread(target=self.process_recording, daemon=True).start()
    
    def process_recording(self):
        """Process the recording with selected models"""
        try:
            if not self.audio_data:
                rumps.notification("Error", "", "No audio recorded", sound=True)
                self.reset_ui()
                return
            
            # Save audio
            audio_array = np.concatenate(self.audio_data, axis=0)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wav_path = self.current_session_dir / "recording.wav"
            wavfile.write(str(wav_path), self.sample_rate, audio_int16)
            
            duration = len(audio_int16) / self.sample_rate
            
            # Get model paths
            whisper_model = self.whisper_models_dir / WHISPER_MODELS[self.current_whisper]["file"]
            llm_model = self.models_dir / LLAMA_MODELS[self.current_llm]["file"]
            llm_template = LLAMA_MODELS[self.current_llm]["template"]
            
            # Step 1: Transcribe (use notification instead of direct UI update)
            rumps.notification("Processing", "Step 1/3", "🎤 Transcribing audio...", sound=True)
            transcript_path = self.current_session_dir / "transcript.txt"
            
            whisper_cmd = [
                str(self.whisper_cli),
                "-m", str(whisper_model),
                "-f", str(wav_path),
                "-otxt",
                "-of", str(self.current_session_dir / "transcript")
            ]
            
            result = subprocess.run(whisper_cmd, capture_output=True, text=True)
            
            if not transcript_path.exists():
                raise Exception("Transcription failed")
            
            transcript = transcript_path.read_text().strip()
            if not transcript:
                transcript = "(No speech detected)"
                transcript_path.write_text(transcript)
            
            # Step 2: Summarize (use notification - thread-safe)
            rumps.notification("Processing", "Step 2/3", f"🧠 Summarizing with {self.current_llm.split('(')[0].strip()}...", sound=True)
            
            max_chars = 4000
            truncated = transcript[:max_chars] + "..." if len(transcript) > max_chars else transcript
            
            summary_system = "You are a helpful assistant that creates concise meeting summaries."
            summary_user = f"Summarize this meeting transcript in 3-5 bullet points:\n\n{truncated}"
            summary_prompt = get_prompt_template(llm_template, summary_system, summary_user)
            
            llama_cmd = [
                str(self.llama_cli),
                "-m", str(llm_model),
                "-p", summary_prompt,
                "-n", "400",
                "--temp", "0.3",
                "--no-display-prompt",  # Don't echo the prompt
                "-e"  # Enable escape sequences (prevents interactive mode)
            ]
            
            # Use stdin to prevent interactive mode from waiting for input
            result = subprocess.run(llama_cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            summary = self.clean_output(result.stdout, llm_template)
            
            summary_path = self.current_session_dir / "summary.txt"
            summary_path.write_text(summary)
            
            # Step 3: Extract action items (use notification - thread-safe)
            rumps.notification("Processing", "Step 3/3", "✅ Extracting action items...", sound=True)
            
            todo_system = "You are a helpful assistant that extracts action items from meetings."
            todo_user = f"Extract all action items as a numbered list. If none exist, say so.\n\n{truncated}"
            todo_prompt = get_prompt_template(llm_template, todo_system, todo_user)
            
            llama_cmd[3] = todo_prompt
            result = subprocess.run(llama_cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            todos = self.clean_output(result.stdout, llm_template)
            
            todo_path = self.current_session_dir / "todos.txt"
            todo_path.write_text(todos)
            
            # Create full report
            report = f"""MEETING RECORDING REPORT
{'='*50}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Duration: {duration:.1f} seconds

MODELS USED (via GGML on Apple Silicon):
- Whisper: {self.current_whisper}
- LLM: {self.current_llm}

{'='*50}
TRANSCRIPT
{'='*50}
{transcript}

{'='*50}
SUMMARY
{'='*50}
{summary}

{'='*50}
ACTION ITEMS
{'='*50}
{todos}

{'='*50}
Powered by GGML - Zero cloud, zero cost!
"""
            report_path = self.current_session_dir / "full_report.txt"
            report_path.write_text(report)
            
            rumps.notification(
                "Processing Complete! ✅",
                f"Used {self.current_llm.split('(')[0].strip()}",
                f"Saved to: {self.current_session_dir.name}",
                sound=True
            )
            
        except Exception as e:
            rumps.notification("Error", "", str(e)[:100], sound=True)
            print(f"Error: {e}")
        
        finally:
            self.reset_ui()
    
    def clean_output(self, output, template):
        """Clean LLM output"""
        lines = output.strip().split('\n')
        result_lines = []
        capture = False
        
        for line in lines:
            # Start capturing after the assistant marker
            if 'assistant' in line.lower() or '<|im_start|>assistant' in line:
                capture = True
                continue
            if '[/INST]' in line:
                capture = True
                continue
            if capture:
                # Skip metadata lines
                if line.startswith(('> EOF', 'common_perf', 'llama_', 'ggml_')):
                    continue
                if '<|im_end|>' in line or '<|eot_id|>' in line:
                    continue
                result_lines.append(line)
        
        return '\n'.join(result_lines).strip() if result_lines else output.strip()
    
    def reset_ui(self):
        """Reset to ready state - dispatches to main thread"""
        def do_reset():
            self.title = "🎙️"
            self.record_button.title = "▶️ Start Recording"
            self.status_item.title = "Status: Ready"
        
        # Use PyObjC to schedule on main thread
        AppHelper.callAfter(do_reset)
    
    def open_recordings(self, sender):
        """Open recordings folder"""
        subprocess.run(["open", str(self.recordings_dir)])


if __name__ == "__main__":
    app = MeetingRecorderProApp()
    
    # Check for required files
    missing = []
    if not app.whisper_cli.exists():
        missing.append("whisper-cli not found")
    if not app.llama_cli.exists():
        missing.append("llama-cli not found")
    
    # Check for at least one model
    has_whisper = any((app.whisper_models_dir / m["file"]).exists() for m in WHISPER_MODELS.values())
    has_llm = any((app.models_dir / m["file"]).exists() for m in LLAMA_MODELS.values())
    
    if not has_whisper:
        missing.append("No Whisper models found")
    if not has_llm:
        missing.append("No LLM models found")
    
    if missing:
        print("Missing requirements:")
        for m in missing:
            print(f"  - {m}")
        exit(1)
    
    print("=" * 50)
    print("Meeting Recorder Pro - GGML Edition")
    print("=" * 50)
    print(f"Recordings: {app.recordings_dir}")
    print(f"Default LLM: {app.current_llm}")
    print(f"Default Whisper: {app.current_whisper}")
    print("")
    print("Look for the 🎙️ icon in your menu bar!")
    print("=" * 50)
    
    app.run()
