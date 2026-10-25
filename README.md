# Infrastructure Forensics: Multi-Sensor Road Repair Prioritization

Minor Project, B.Tech CSE, Punjab Engineering College, Chandigarh (Session 2026–27)
Mentor: **Dr. Padmavati**

| Member | SID | Module |
|---|---|---|
| Keshav Garg | 24103071 | Risk scoring, repair cost model, budget optimizer, dashboard |
| Sehaj Bajaj | 24103095 | Sensor rig, ESP32 firmware, sensor–video synchronisation |
| Praveer Singh | 24103101 | CNN training and road damage detection |

---

## Problem

Existing systems either **detect** road damage or **plan repairs** assuming condition data already exists. No low-cost system links multi-sensor detection to **budget-constrained repair prioritization** for Indian cities. This project builds that link for selected roads in Chandigarh.

## Pipeline

```
Sense  →  Detect  →  Score  →  Cost  →  Decide
```

| Step | What happens | Tools |
|---|---|---|
| **Sense** | A bike-mounted ESP32 rig logs vibration, GPS and road-surface distance at 100 Hz while a phone records video | ESP32, MPU6050, NEO-6M, HC-SR04, microSD |
| **Detect** | A YOLO model finds cracks and potholes in the video; vibration spikes confirm them | Ultralytics YOLO, Python |
| **Score** | Each road is split into 50 m cells, and every cell gets a risk score | XGBoost, SHAP |
| **Cost** | Repair cost = damaged area × PWD/CPWD Schedule of Rates | Python |
| **Decide** | Choose the set of cells that removes the most risk within a given budget (0/1 knapsack), compared with greedy and random | OR-Tools, Streamlit dashboard |

## Repository structure

```
infrastructure-forensics/
├── README.md
├── cnn/
│   ├── infra_forensics_cnn_training.ipynb   # Kaggle training notebook
│   ├── run_detection.py                     # Runs the trained model on a road video
│   └── weights/infra_forensics_best.pt      # Trained model weights
├── results/
│   ├── metrics.json                         # Test-set metrics (all countries + India only)
│   ├── results.png                          # Training curves
│   ├── confusion_matrix.png
│   └── samples/                             # Example detections
└── firmware/                                # Arduino sketches for the ESP32
    ├── 01_mpu_test/      # Checks the MPU6050 accelerometer/gyroscope
    ├── 02_gps_test/      # Checks the NEO-6M GPS fix
    ├── 03_ultrasonic_test/  # Checks the HC-SR04 distance reading
    ├── 04_sd_test/       # Checks microSD write
    └── ride_logger/      # Main logger used during survey rides
```

---

## 1. CNN: road damage detection (Praveer Singh)

- **Model:** Ultralytics YOLO26-s, pretrained on COCO, fine-tuned for road damage
- **Dataset:** RDD2022 (Road Damage Dataset 2022, Arya et al.), with images from India, Japan, the Czech Republic, Norway, the USA and China
- **Classes (IRC-style):** 0 longitudinal crack, 1 transverse crack, 2 alligator crack, 3 other corruption, 4 pothole
- **Label format:** YOLO, one `.txt` per image, one line per object: `class x_center y_center width height` (all normalised to 0–1)
- **Training:** Kaggle, NVIDIA T4 GPU, image size 640, batch 32, 8-hour budget (56 epochs), early-stopping patience 25

### Results (RDD2022 validation split)

| Metric | All countries | India only |
|---|---|---|
| mAP@50 | **0.639** | 0.437 |
| mAP@50-95 | 0.351 | 0.210 |
| Precision | 0.690 | 0.455 |
| Recall | 0.587 | 0.480 |
| Inference time | 9.9 ms/image (~86 FPS) | 10.7 ms/image |

| Class | AP@50 (all) |
|---|---|
| Longitudinal crack | 0.594 |
| Transverse crack | 0.583 |
| Alligator crack | 0.688 |
| Other corruption | 0.806 |
| Pothole | 0.525 |

The model scores lower on Indian roads, so it will be fine-tuned on frames from our own Chandigarh survey videos.

### Run detection on a video

```bash
pip install ultralytics opencv-python
python cnn/run_detection.py --video road1.mp4 --model cnn/weights/infra_forensics_best.pt --fps 5 --conf 0.25
```

Output: `detections_road1.csv` (frame, timestamp, class, confidence, bounding box) and `samples_road1/` (annotated frames).

---

## 2. Sensor rig and firmware (Sehaj Bajaj)

| Sensor | Measures | Connection (ESP32 pin) |
|---|---|---|
| MPU6050 | Vibration (acceleration ±8 g, rotation) | I²C: SDA 21, SCL 22, 3.3 V |
| NEO-6M GPS | Location, speed, UTC time | UART2: RX 16, TX 17, 9600 baud, 3.3 V |
| HC-SR04 | Distance to road surface (pothole depth) | TRIG 27, ECHO 26 via 1 kΩ/2 kΩ divider, 5 V |
| microSD module | Stores the ride log | SPI: CS 5, SCK 18, MISO 19, MOSI 23, 5 V |

**`ride_logger.ino`**
- Samples the IMU at a fixed **100 Hz** (drift-free timing)
- Pings the ultrasonic sensor every 100 ms, timing the echo with an **interrupt**, so the loop never blocks
- Writes one CSV row per sample to `/ride_XXX.csv` on the SD card:
  `millis, gps_valid, gps_date, gps_time_utc, gps_age_ms, lat, lng, speed_kmh, sats, ax, ay, az, gx, gy, gz, dist_cm`
- On-board LED: solid = GPS fix, blinking = waiting for fix; error blinks: 2 = MPU, 3 = SD, 4 = file

**Upload:** Arduino IDE → ESP32 board package → libraries `TinyGPSPlus`, `Adafruit MPU6050` → run the four test sketches one by one, then upload `ride_logger`.

**Sensor–video sync:** At the start and end of each ride, the box is tapped 3 times. The taps show up as spikes in the accelerometer log and as clicks in the video's audio, which gives the time offset between the two recordings.

---

## 3. Risk, cost and optimizer (Keshav Garg)

- Each road is split into **50 m cells** (OpenStreetMap geometry)
- Per-cell features: detection counts by class, damaged area, vibration peaks and RMS, ultrasonic depth
- **Risk model:** XGBoost, validated by holding out whole roads, explained with SHAP
- **Cost:** damaged area × PWD/CPWD Schedule of Rates
- **Optimizer:** 0/1 knapsack (OR-Tools) maximising total risk removed within a budget, compared with greedy and random selection
- **Dashboard:** Streamlit map of cells coloured by risk, with a budget slider and per-cell evidence

---

## Status (mid-term)

| Module | Status |
|---|---|
| Literature review | Completed |
| CNN training and evaluation | Completed |
| Sensor rig and firmware | In progress (wiring being finalised) |
| Survey rides, risk model, optimizer, dashboard | Planned |

## Key references

1. Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*, 2022
2. Eriksson et al., *The Pothole Patrol: Using a Mobile Sensor Network for Road Surface Monitoring*, ACM MobiSys 2008
3. Mohan et al., *Nericell: Rich Monitoring of Road and Traffic Conditions using Mobile Smartphones*, ACM SenSys 2008
4. Mednis et al., *Real Time Pothole Detection using Android Smartphones with Accelerometers*, IEEE DCOSS 2011
5. IRC:82, *Code of Practice for Maintenance of Bituminous Roads*, Indian Roads Congress
