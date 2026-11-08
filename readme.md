# Virtual Paint

A real-time drawing app that uses your webcam to track colored markers and paint their movement on screen.

## Description

Virtual Paint turns any colored marker into a digital pen. The app reads video from your webcam, finds each marker by its color, and draws a line following its path. The video is mirrored so the drawing moves the way your hand does. Skin tones are filtered out so your hand does not paint by accident.

## Features

- Live drawing with three marker colors: green, blue, and magenta
- Mirrored camera view for natural movement
- Skin-tone rejection to reduce false detections
- Smoothed strokes with jump rejection and brief dropout handling
- Adjustable color ranges for different lighting and marker shades
- Mask debug view to see what each color detects
- Capped point storage to keep performance steady
- Keyboard controls for clearing the canvas and quitting

## Controls

| Key | Action        |
|-----|---------------|
| `c` | Clear canvas  |
| `m` | Toggle masks  |
| `q` | Quit          |

## Tools and Technologies

- Python 3
- OpenCV (`opencv-python`)
- NumPy
- Webcam

## Installation

```bash
pip install opencv-python numpy
python VirtualPaintProject.py
```
