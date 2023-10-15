# Facial Recognition Attendance System


## Introduction

This project, which was inspired by Zain Shahbaz's YouTube videos (@iamzainshahbaz), is designed for performing facial recognition-based attendance tracking using Python and various libraries. It is intended for educational institutions, allowing you to recognize students' faces and record their attendance efficiently.

## Project Overview

The project includes the following components:

### File Structure

This project's file structure is as follows:

1. `attendance.csv`: This CSV file is used to store the attendance records of recognized students. It records the students who are present on a given day, along with their timestamps. It also includes the names of absent students.

2. `Database` Folder: The `Database` folder contains the images of students organized into subfolders, typically by class. These images are used for face recognition during the attendance tracking process.

3. `requirements.txt`: This file lists the Python libraries and their versions required to run the project successfully. It's a standard practice to include this file to ensure that others can easily set up their environment with the necessary dependencies.

4. `face_recognition.py`: This Python script performs the core functionality of the attendance tracking system. It uses the DeepFace library for face recognition. It loads a pre-trained liveness model used to distinguish between real and fake faces, captures video from a camera source, detects and recognizes faces in the captured frames, records the attendance of recognized students in the `attendance.csv` file with timestamps, and displays recognized students on the screen, indicating whether their faces are considered real or fake. If a student is not recognized, they are labeled as "Unknown."

5. `face_recognition-checkpoint.py`: This script is an alternative version of `face_recognition.py`. It performs similar functions, including capturing video, detecting and recognizing faces, and distinguishing real and fake faces. However, it's organized differently and contains commented-out code that can be useful for reference or experimentation.

6. `liveness.model`: This file contains a pre-trained model for detecting liveness. It is used to determine whether the captured face is a real face or a photograph. The model is essential for ensuring the accuracy of attendance records.

### Installation

To install and run the project, follow these steps:

- Install the required libraries listed in `requirements.txt`. You can do this using `pip install -r requirements.txt`.

- Organize your student images in the `Database` folder. Each subfolder should correspond to a class or group, containing images of the students.

- Run the `face_recognition.py` script. It will capture video from your camera source and begin recognizing faces.

- Recognized students will be recorded in the `attendance.csv` file with timestamps, and their names will be displayed on the screen.

- The script will also detect absent students and record them in the `attendance.csv` file.

### Customization

Feel free to customize the code or the way you organize the `Database` folder to suit your specific use case and educational institution's needs.

### Dependencies

The project depends on the following libraries:

- `numpy==1.19.5`
- `tensorflow==2.5.0`
- `opencv-python==4.5.3.56`
- `deepface==0.0.68`
- `pandas==1.1.5`

Ensure you have these libraries installed in your environment to run the project successfully.

### License

You are free to use, modify, and distribute the code. However, I kindly request that you use it in an ethical manner, respecting privacy and legal regulations, and ensuring the technology is not misused for harmful or unethical purposes.

### Available Models

You can choose different face recognition models by referring to [DeepFace](https://github.com/serengil/deepface). Here are some popular models and their performance scores:

| Model         | LFW Score | YTF Score |
|---------------|-----------|-----------|
| Facenet512    | 99.65%    | -         |
| SFace         | 99.60%    | -         |
| ArcFace       | 99.41%    | -         |
| Dlib          | 99.38%    | -         |
| Facenet       | 99.20%    | -         |
| VGG-Face      | 98.78%    | 97.40%    |
| Human-beings  | 97.53%    | -         |
| OpenFace      | 93.80%    | -         |
| DeepID        | -         | 97.05%    |

