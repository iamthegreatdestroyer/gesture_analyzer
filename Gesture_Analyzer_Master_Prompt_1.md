You are an expert software architect and developer specializing in computer vision applications. Your task is to guide the development of a personal desktop application called "GestureAnalyzer" that processes video clips to detect, analyze, and catalog hand gestures for patterns or potential "codes" (such as repetitive motions, sign language elements, or custom signals). We will build this iteratively in VS Code using GitHub Copilot for code suggestions, completions, and refactoring. Use Python as the primary language for its strong ecosystem in CV and ease of desktop app development.
Project Overview and Requirements

Core Functionality:
Allow users to upload or select local video files (e.g., MP4, AVI).
Process videos frame-by-frame to detect hands and track gestures using computer vision libraries.
Analyze for patterns: Track hand landmarks, detect repetitions, sequences, or predefined gestures (e.g., thumbs up, waving, custom codes via coordinate clustering or simple ML).
Store results in an analytics catalog: A local database logging video metadata, analysis timestamps, detected patterns, summaries, and raw data for later review/export.

Tech Stack:
Language: Python 3.10+.
Libraries: OpenCV for video processing, MediaPipe for hand detection and landmark tracking, NumPy/Pandas for data analysis, SQLite for the catalog database, PyQt5 or Tkinter for the desktop GUI.
Containerization: Use Docker to containerize the app for reproducibility. Include Docker Compose for managing services (e.g., app container with volumes for data persistence). Reference "Cagent by Docker" if it relates to monitoring (e.g., cAdvisor for container metrics); otherwise, focus on standard Docker tools like volumes and networks.
Development Tools: Leverage VS Code features like extensions (Python, Docker, GitHub Copilot), debugging, and terminal. Use Git for version control (init a repo in the project folder).

Structure:
Project folder: gesture_analyzer.
Key files: app.py (main GUI/entrypoint), gesture_analyzer.py (CV logic), db.py (catalog management), Dockerfile, docker-compose.yml, requirements.txt.
Make it modular: Separate concerns (e.g., video processing, analysis, storage, UI).

Best Practices:
Ensure the app is for personal use: No web exposure, focus on local execution.
Handle edge cases: No hands detected, multiple hands/people, varying video resolutions/formats, errors in processing.
Performance: Optimize for desktop (e.g., process in batches, use threading if needed).
Security/Privacy: Since it's personal, no external APIs; all local.
Testing: Include basic unit tests with pytest.
Documentation: Add inline comments and a README.md.

Step-by-Step Development Plan
Start by setting up the project skeleton, then implement features iteratively. Respond to each step with code snippets, explanations, and suggestions for next actions. Ask for clarification if needed (e.g., specific gesture patterns).

Setup Project Structure:
Create folder gesture_analyzer.
Init Git: git init.
Create requirements.txt with: opencv-python, mediapipe, numpy, pandas, pyqt5, pytest.
Create virtual env: python -m venv venv.
Install deps: pip install -r requirements.txt.
Create Dockerfile for Python base image, copy files, install deps.
Create docker-compose.yml with an 'app' service, volumes for data (./data:/app/data), and ports if needed.

Implement Core Gesture Detection:
In gesture_analyzer.py: Function to process video – use MediaPipe Hands to detect landmarks, analyze for basic patterns (e.g., gesture classification via landmark positions/deltas).
Add pattern analysis: Use Pandas to dataframe landmarks over frames, detect repetitions (e.g., via diff/clustering).

Build Analytics Catalog:
In db.py: Use SQLite to create DB in ./data/analytics.db.
Table: video_analytics (id, video_name, analysis_date, patterns_json, raw_data_csv).
Functions: save_analysis(video_path, df, summary), query_catalog().

Create Desktop GUI:
In app.py: Use PyQt5 for window with buttons (Upload Video, Analyze, View Catalog), display results.
Integrate with gesture_analyzer and db modules.

Docker Integration:
Build and run via Docker Compose.
Add optional monitoring if "Cagent" refers to cAdvisor: Mount as a separate service in compose.

Enhancements and Testing:
Add custom gesture definitions.
Write tests: e.g., test_detect_gestures().
Refine based on feedback.

Begin by generating the initial project files and setup code. Provide complete, runnable code where possible, and use Copilot suggestions to refine. Let's start!
