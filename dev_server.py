#!/usr/bin/env python3
"""
Development server with enhanced file watching capabilities.
Automatically reloads when any Python, HTML, CSS, or JS files change.
"""

import os
import sys
from flask import Flask
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

class ReloadHandler(FileSystemEventHandler):
    """Custom file system event handler for auto-reload"""
    
    def __init__(self):
        self.last_reload = 0
        
    def on_modified(self, event):
        # Only reload for specific file types and avoid rapid successive reloads
        if (not event.is_directory and 
            event.src_path.endswith(('.py', '.html', '.css', '.js')) and
            time.time() - self.last_reload > 1):  # Debounce for 1 second
            
            print(f"🔄 File changed: {event.src_path}")
            print("📡 Auto-reload triggered by Flask...")
            self.last_reload = time.time()

def setup_file_watcher():
    """Set up file system watcher for better feedback"""
    event_handler = ReloadHandler()
    observer = Observer()
    
    # Watch current directory and subdirectories
    watch_paths = ['.', 'templates', 'static', 'src']
    
    for path in watch_paths:
        if os.path.exists(path):
            observer.schedule(event_handler, path, recursive=True)
            print(f"👁️  Watching: {os.path.abspath(path)}")
    
    observer.start()
    return observer

def run_dev_server():
    """Run the development server with enhanced watching"""
    print("🚀 Starting development server with auto-reload...")
    print("🔧 Enhanced file watching enabled")
    print("📍 Server will be available at: http://127.0.0.1:3000")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Start file watcher in a separate thread
    observer = setup_file_watcher()
    
    try:
        # Run Flask app with enhanced configuration
        app.run(
            host="0.0.0.0",
            port=3000,
            debug=True,
            use_reloader=True,
            use_debugger=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Stopping development server...")
        observer.stop()
    
    observer.join()

if __name__ == '__main__':
    run_dev_server()