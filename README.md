# Driver Drowsiness Detection 🚗

A real-time deep learning system that monitors a driver through a camera and detects signs of drowsiness to help prevent fatigue-related accidents.

> **Status:** 🚧 In progress. The project is currently in the planning phase.

## 👥 Team Members

* **Manar Abu Irsheid** (منار أبو ارشيد)
* **Sondos Abushunar** (سندس أبو شنار)
* **Noor Al-Sherman** (نور الشرمان)
* **Hala Abu Shindi** (هالة أبو شندي)

## Overview

The system is designed to track three key indicators of fatigue:

- **Eye closure:** detecting prolonged or frequent eye closing.
- **Yawning:** detecting yawns as an early sign of tiredness.
- **Head pose:** monitoring head direction and nodding.

When drowsiness is detected, the system will trigger an alert to wake the driver and draw their attention back to the road.

## Planned Approach

The project will combine three components:

- **MediaPipe:** extract facial landmarks (eyes, mouth, head orientation) from the video stream in real time.
- **CNN:** classify the state of the eyes and mouth (open/closed, yawning/not yawning).
- **LSTM:** model the sequence of frames over time, so decisions are based on behavior across several frames rather than a single image.

## Planned Features

- Real-time detection from a webcam
- Multi-signal analysis (eyes, yawning, head pose)
- Temporal modeling for more reliable predictions
- Automatic alert when drowsiness is detected

## Roadmap

- [ ] Collect and prepare the dataset
- [ ] Extract facial landmarks with MediaPipe
- [ ] Train the CNN model
- [ ] Train the LSTM model
- [ ] Build the real-time detection pipeline
- [ ] Evaluate and document results
