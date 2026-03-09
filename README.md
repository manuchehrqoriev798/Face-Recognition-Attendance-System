# Face Recognition Attendance System

![fra](https://github.com/manuchehrqoriev798/Face-Recognition-Attendance-System/assets/112572372/ffb4a653-d734-4419-826c-603b9da250cf)

A Python-based attendance system that uses **face recognition** and **liveness detection** to mark students present or absent. It records attendance in a CSV file with timestamps and supports organizing students by class.

Inspired by [Zain Shahbaz](https://www.youtube.com/@iamzainshahbaz) (YouTube). Built for schools and institutions that want automatic, camera-based attendance.

---

## What This Project Does

- **Recognizes faces** from your webcam using a reference photo database ([DeepFace](https://github.com/serengil/deepface), Facenet512).
- **Liveness check** so photos/screens cannot be used to fake attendance (pre-trained `liveness.model`).
- **Writes attendance** to `attendance.csv`: present students with time, then a list of absent students.
- **Organized by class**: database layout is `Database/<Class>/<StudentName>/` with one or more photos per student.

---

## Prerequisites

- **Python 3.7–3.9** (TensorFlow 2.5 works best in this range)
- **Webcam**
- **pip**

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/manuchehrqoriev798/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up the student database

Create the folder structure and add one or more photos per student (clear face, front-facing):

```
Database/
├── ClassA/
│   ├── Alice/
│   │   └── photo1.jpg
│   └── Bob/
│       └── photo1.jpg
└── ClassB/
    └── Carol/
        └── photo1.jpg
```

Create the folders (example for one class and two students):

```bash
mkdir -p Database/ClassA/Alice
mkdir -p Database/ClassA/Bob
```

Then copy or place face photos into each student folder (e.g. `Database/ClassA/Alice/photo1.jpg`).

### 5. Run the attendance script

```bash
python face_recognition.py
```

- Point the camera at students. Recognized faces are marked and logged to `attendance.csv`.
- Press **`q`** to quit. After quitting, absent students are appended to the same CSV.

---

## Files in this repo

**Which script to run:** use **`face_recognition.py`**. It is the full, up-to-date version. The other script is an older checkpoint and does less.

### Scripts (Python)

| File | Purpose | When to use |
|------|---------|-------------|
| **`face_recognition.py`** | **Main script.** Runs the camera, recognizes faces, runs liveness check, writes present students (with time) to CSV, then on exit writes the absent list. Expects database layout **`Database/<Class>/<StudentName>/`** with photos inside each student folder. Each student is logged only once; absent list is built from all students in `Database/` after you press `q`. | **Use this one.** |
| **`face_recogntion-checkpoin.py`** | **Older / backup version.** Same idea (camera + DeepFace + liveness) but: (1) does **not** use a Class/Student folder structure — it only uses one folder level (student name = folder name); (2) writes to CSV **every time** a face is seen as "Real" (so duplicate rows for the same person); (3) writes only the **date**, not time; (4) **does not** write an absent list when you quit. Kept for reference or rollback only. | Do **not** use unless you need the old behavior. |

So: **run only `face_recognition.py`**. The checkpoint file does not "re-run" the main script; it is a separate, simpler version that was saved before adding classes, timestamps, and the absent list.

### Other files

| File or folder | Purpose |
|----------------|--------|
| **`liveness.model`** | Pre-trained Keras model used by both scripts. Detects real face vs photo/screen so attendance cannot be faked with a picture. |
| **`attendance.csv`** | **Output.** Created/updated by `face_recognition.py`: first a header and "Present Students list:", then one row per present student with number, name, class, and time; after you press `q`, "Absent Student list:" and one row per absent student (number, name, class). |
| **`Database/`** | **You create this.** Put student photos in **`Database/<Class>/<StudentName>/`** (e.g. `Database/ClassA/Alice/photo1.jpg`). Required for `face_recognition.py`. |
| **`requirements.txt`** | List of Python packages and versions. Use with `pip install -r requirements.txt` for installation. |
| **`requarements.txt`** | Same content as `requirements.txt` (typo in name). You can ignore it and use `requirements.txt`. |

---

## Configuration

- **Camera index**  
  In `face_recognition.py`, the line `cap = cv2.VideoCapture(1)` uses camera `1`. If you have a single webcam, change it to `0`:

  ```python
  cap = cv2.VideoCapture(0)
  ```

- **Face recognition model**  
  The script uses `model_name='Facenet512'`. You can switch to other [DeepFace models](https://github.com/serengil/deepface) (e.g. `SFace`, `ArcFace`, `VGG-Face`) by changing that argument in the `DeepFace.find(...)` call.

---

## Dependencies (from requirements.txt)

| Package | Version |
|---------|---------|
| numpy | 1.19.5 |
| tensorflow | 2.5.0 |
| opencv-python | 4.5.3.56 |
| deepface | 0.0.68 |
| pandas | 1.1.5 |

---

## DeepFace model comparison (reference)

Benchmark scores from the [DeepFace](https://github.com/serengil/deepface) library; measured on LFW via [benchmarks](https://github.com/serengil/deepface/tree/master/benchmarks).

| Model        | Measured Score | Declared Score |
| ------------ | -------------- | -------------- |
| Facenet512   | 98.4%          | 99.6%          |
| Human-beings | 97.5%          | 97.5%          |
| Facenet      | 97.4%          | 99.2%          |
| Dlib         | 96.8%          | 99.3%          |
| VGG-Face     | 96.7%          | 98.9%          |
| ArcFace      | 96.7%          | 99.5%          |
| GhostFaceNet | 93.3%          | 99.7%          |
| SFace        | 93.0%          | 99.5%          |
| OpenFace     | 78.7%          | 92.9%          |
| DeepFace     | 69.0%          | 97.3%          |
| DeepID       | 66.5%          | 97.4%          |

---

## License and use

You may use, modify, and distribute this code. Please use it in an ethical way, with respect for privacy and local laws, and do not use it for harmful or unethical purposes.
