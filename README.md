# In-Cabin Posture Classification and Pose Reconstruction Using Pressure Sensor Fusion

A privacy-preserving in-cabin monitoring system that uses pressure sensors for high-accuracy posture classification and camera-less 2D pose reconstruction.

This project provides a robust, non-visual solution for monitoring vehicle occupants. By leveraging a suite of pressure sensors, it can accurately classify a person's posture and reconstruct their pose in real-time, all without using cameras, thus ensuring user privacy.

---

## ✨ Key Features
- **High-Accuracy Posture Classification**: A Two-Stream CNN achieves 99.8% accuracy in classifying good, bad, and drowsy postures using a high-resolution pressure grid and discrete sensors.  
- **Novel Pose Reconstruction**: A LightGBM model reconstructs a 2D stick-figure of the occupant with 98% accuracy.  
- **Privacy-Preserving by Design**: The entire system is camera-less, relying on pressure, radar, and IR sensors.  
- **Efficient Reconstruction**: The pose reconstruction model is computationally lightweight, using data from only 8 discrete sensors.  
- **Real-Time Performance**: The reconstruction code is optimized for live demonstration with data smoothing and fluid animations.  

---

## 💾 Dataset

The dataset used for training and evaluating the models in this project is available through the following google drive link. 
Link to the dataset : https://drive.google.com/drive/folders/1KjwUVbRw_ypXqomj9mjPc3Kw55bs-yVS

## 🛠️ Hardware Requirements
The system was built and tested using the following hardware:

**Pressure Sensors:**
- 1× 15×15 Velostat pressure grid (seat cushion)  
- 8× discrete Velostat pressure sensors (seatback and seatbelt)  

**Physiological Sensor:**
- 1× 60 GHz mmWave radar module (for heart rate)  

**Head Tracking:**
- 2× Infrared (IR) proximity sensors (headrest)  

**Processing Unit:**
- A central microcontroller (e.g., Arduino or ESP32) to aggregate and transmit sensor data via Serial/USB.  

