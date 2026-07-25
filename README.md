# VisionFit

VisionFit is a computer vision-based fitness assistant built using Python, OpenCV and MediaPipe. It tracks body movements through a webcam and counts push-up repetitions in real time. The project also supports offline voice commands, allowing users to start, pause and reset a workout without touching the keyboard.

This project was built to explore pose estimation, human movement tracking and voice-controlled interaction using lightweight machine learning models.

## Features

- Real-time push-up repetition counter
- Pose estimation using MediaPipe
- Elbow angle calculation for accurate rep detection
- Offline voice commands using Vosk
- Start, pause and reset workouts using voice
- Live pose landmark visualisation
- Works completely offline

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Vosk Speech Recognition
- SoundDevice

## Installation

Clone the repository:

```bash
git clone https://github.com/varen1512/VisionFit.git
cd VisionFit
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the Vosk English model from:

https://alphacephei.com/vosk/models

Download:

```
vosk-model-small-en-us-0.15
```

Extract the folder into the project directory.

Run the application:

```bash
python VisionFit.py
```

## Voice Commands

| Command | Action |
|---------|--------|
| Start | Starts or resumes the workout |
| Pause | Pauses the workout while keeping the current rep count |
| Reset | Resets the repetition counter |

## Current Limitations

- Currently supports push-up detection only
- Designed for a single person in front of the camera
- Voice recognition is limited to a few predefined commands

## Future Work

- Support for multiple exercises
- Posture correction feedback
- Workout history and statistics
- Calorie estimation
- Desktop application
- Exercise recommendation system

## Project Structure

```
VisionFit/
│
├── VisionFit.py
├── requirements.txt
├── README.md
├── .gitignore
└── vosk-model-small-en-us-0.15/   # Download separately
```

## Author

**Varen**

GitHub: https://github.com/varen1512