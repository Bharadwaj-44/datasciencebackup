#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Startup configuration for DSA kernel - EC2 Optimized

This file runs automatically when the Jupyter kernel starts.

It ONLY sets environment variables to avoid hanging on headless servers.

"""
 
import os

import sys
 
print("=" * 70)

print("🔧 DSA KERNEL STARTUP: Configuring environment for headless EC2...")

print("=" * 70)
 
# ✅ CRITICAL: Set matplotlib backend BEFORE any imports

# These environment variables are checked when matplotlib imports later

os.environ['MPLBACKEND'] = 'Agg'  # Non-GUI backend (file-based)

os.environ['DISPLAY'] = ''  # No display available (headless)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Qt offscreen mode
 
print("✅ Environment variables set:")

print("   • MPLBACKEND = 'Agg' (saves to files, no GUI)")

print("   • DISPLAY = '' (headless mode)")

print("   • QT_QPA_PLATFORM = 'offscreen'")

print()

print("✅ Kernel environment configured successfully!")

print("   Matplotlib will automatically use Agg backend when imported.")

print("   Charts will be saved as PNG files and displayed in Gradio.")

print("=" * 70)
 
