# AI Waste Detection System

## Overview
AI Waste Detection System is a Django-based web application that uses YOLOv8 for waste detection and classification. The system detects waste objects from uploaded images and classifies them as Biodegradable or Non-Biodegradable.

## Features
- Upload waste images through a web interface
- Detect multiple waste objects in a single image
- Draw bounding boxes around detected objects
- Display confidence scores for each detection
- Classify waste as:
  - Biodegradable
  - Non-Biodegradable
- Detection history management

## Technologies Used
- Python
- Django
- YOLOv8 (Ultralytics)
- OpenCV
- HTML
- CSS
- Bootstrap
- SQLite

## Project Structure

```
waste_project/
│
├── detector/
├── templates/
├── static/
├── media/
├── waste_project/
├── manage.py
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/MaluKrish27/Waste-Detection-System.git
```

### Navigate to Project

```bash
cd waste_project
```

### Create Virtual Environment

```bash
python -m venv env
```

### Activate Environment

```bash
env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Migrations

```bash
python manage.py migrate
```

### Start Server

```bash
python manage.py runserver
```

## Usage

1. Open the application in a browser.
2. Upload a waste image.
3. Click Detect.
4. View detected waste objects, confidence scores, and waste classification results.

## Business Use Cases

- Smart Waste Segregation
- Municipal Waste Management
- Recycling Centers
- Smart Cities
- Environmental Monitoring

## Future Enhancements

- Real-time webcam detection
- Mobile application integration
- More waste categories
- Cloud deployment
- Waste analytics dashboard

## Author

**Malavika Krishnan**

MCA Project – AI Waste Detection System using YOLOv8 and Django
