<br>
<h1 align="center">DroneInfraCV – AI-Powered Infrastructure Inspection System</h1>
<br>
<br>
<h2 id="deployment" align="center">Deployment</h2>
<br>

<p align="center">
  <a href="https://aidroneinspection-fxwvr89b8uikxghshywfzj.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Launch_Live_Demo-Streamlit-orange?style=for-the-badge&logo=streamlit" />
  </a>
</p>
<p align="center">
An end-to-end computer vision pipeline for automated detection of infrastructure defects using drone imagery.  
The system leverages deep learning to identify cracks, potholes, and rust with high accuracy, enabling scalable and intelligent inspection workflows.
</p>


<br>
<h2 id="demo" align="center">Demo</h2>
<br>

<p align="center">
<img src="outputs/output_inspection.gif" alt="Demo Preview" width="600"/>
</p>


<br>
<h2 align="center">Table of Contents</h2>
<br>

- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Architecture](#architecture)  
- [Detection Visualization](#visualization)  
- [Demo](#demo)  
- [Deployment](#deployment)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Evaluation](#evaluation)  
- [Future Work](#future-work)  
- [Contact](#contact)  

<br>
<h2 id="overview" align="center">Overview</h2>
<br>

DroneInfraCV is a unified AI system designed to automate infrastructure inspection using drone-based imagery. Traditional inspection methods are slow, costly, and error-prone. This project introduces a scalable deep learning solution capable of detecting multiple defect types within a single pipeline.

The system integrates YOLOv8 for detection and Streamlit for deployment, delivering real-time annotated outputs and structured insights.

<br>
<h2 id="features" align="center">Features</h2>
<br>

- Multi-class detection (cracks, potholes, rust)  
- Real-time image and video inference  
- Clean and intuitive Streamlit interface  
- High-accuracy YOLOv8 model  
- Structured detection outputs  
- Distinct color-coded visualization  

<br>
<h2 id="tech-stack" align="center">Tech Stack</h2>
<br>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.12-blue" />
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red" />
<img src="https://img.shields.io/badge/Streamlit-WebApp-orange" />
<img src="https://img.shields.io/badge/OpenCV-ComputerVision-green" />
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-lightgrey" />
<img src="https://img.shields.io/badge/Pandas-DataAnalysis-blueviolet" />
</p>

<br>
<h2 id="dataset" align="center">Dataset</h2>
<br>

The model is trained on a merged dataset combining multiple sources:

- CrackSeg Dataset  
- Roboflow Pothole Dataset  
- Roboflow Rust Dataset  

All datasets were unified into YOLO format with standardized labels:

- Class 0: Pothole  
- Class 1: Rust  
- Class 2: Crack  

<br>
<h2 id="methodology" align="center">Methodology</h2>
<br>

1. Data collection and merging  
2. Preprocessing and augmentation  
3. YOLOv8 multi-class training  
4. Inference on image/video input  
5. Visualization and reporting  

<br>
<h2 id="architecture" align="center">Architecture</h2>
<br>

<p align="center">
Input → Preprocessing → YOLOv8 Model → Detection Output → Streamlit Visualization
</p>

<br>
<h2 id="visualization" align="center">Detection Visualization</h2>
<br>

Color-coded bounding boxes improve clarity:

- Blue → Pothole  
- Red → Rust  
- Yellow → Crack  

Each detection includes label and confidence score for interpretability.


<p align="center">

</p>

<br>
<h2 id="installation" align="center">Installation</h2>
<br>

```bash
git clone https://github.com/hamaylzahid/AIDroneInspection.git
cd DroneInfraCV

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

<br>
<h2 id="usage" align="center">Usage</h2>
<br>

```bash
streamlit run src/app.py
```

<br>
<h2 id="evaluation" align="center">Evaluation</h2>
<br>

| Metric        | Description                          |
|--------------|--------------------------------------|
| Precision    | Correct positive predictions         |
| Recall       | Detection coverage                   |
| F1 Score     | Balance of precision and recall      |
| IoU          | Overlap accuracy of bounding boxes   |

<br>
<h2 id="future-work" align="center">Future Work</h2>
<br>

- GPS-based defect mapping  
- Real-time drone streaming  
- Edge deployment optimization  
- Dataset expansion  

<br>
<h2 id="contact" align="center">Contact</h2>
<br>

<p align="center">
<a href="https://github.com/hamaylzahid">
<img src="https://img.shields.io/badge/hamaylzahid-black?style=for-the-badge" />
</a>
</p>

<br>

<p align="center">
<img src="https://img.shields.io/github/stars/hamaylzahid/AIDroneInspection?style=for-the-badge" />
<img src="https://img.shields.io/github/forks/hamaylzahid/AIDroneInspection?style=for-the-badge" />
<img src="https://img.shields.io/github/issues/hamaylzahid/AIDroneInspection?style=for-the-badge" />
<img src="https://img.shields.io/github/license/hamaylzahid/AIDroneInspection?style=for-the-badge" />
</p>
