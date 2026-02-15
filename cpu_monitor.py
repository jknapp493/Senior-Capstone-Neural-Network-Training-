#!/usr/bin/env python3
"""
monitor_cpu.py
Monitor Intel 7700K CPU usage in real-time.
"""

import psutil
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

try:
    while True:
        clear_screen()
        print("CPU Monitor\n")
        # Overall CPU usage
        print(f"Total CPU Usage: {psutil.cpu_percent(interval=1)}%")
        
        # Per-core usage
        per_core = psutil.cpu_percent(interval=0.5, percpu=True)
        for i, usage in enumerate(per_core):
            print(f"Core {i}: {usage}%")
        
        # CPU frequency
        freq = psutil.cpu_freq()
        print(f"\nCPU Frequency: {freq.current:.2f} MHz (Min: {freq.min:.2f} MHz, Max: {freq.max:.2f} MHz)")
        
        # Memory info
        mem = psutil.virtual_memory()
        print(f"\nMemory Usage: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)")
        
        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting CPU monitor.")
