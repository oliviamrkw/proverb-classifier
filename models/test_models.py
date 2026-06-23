#!/usr/bin/env python3
import subprocess
import time

models = [
    ("llama3.1", "ollama"),
    ("qwen2.5:7b", "ollama"),
    ("gemma2:9b", "ollama"),
]

print("Testing models...")
for model_name, runner in models:
    try:
        print(f"\n[{model_name}] Testing...")
        result = subprocess.run(
            [runner, "run", model_name, "Say 'OK'"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"  ✓ {model_name} works")
        else:
            print(f"  ✗ {model_name} failed: {result.stderr}")
    except Exception as e:
        print(f"  ✗ {model_name} error: {e}")