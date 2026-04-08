

# AIGVDetPlus: Spatio-Temporal AI Video Detection (Enhanced)

**AIGVDetPlus** is an optimized and deployment-ready implementation of AI-generated video detection based on spatio-temporal anomaly learning.

This project builds upon the original AIGVDet research and extends it with performance optimizations, usability improvements, and a web-based interface for real-world interaction.

---

## I)Overview

The rise of AI-generated (deepfake) videos has created a need for robust detection systems.
The original AIGVDet framework addresses this by analyzing both:

* **Spatial inconsistencies** (visual artifacts in frames)
* **Temporal inconsistencies** (unnatural motion patterns)

**AIGVDetPlus enhances this system by making it faster, more efficient, and easier to use in practical environments.**


---


## II)Original Research Contributions (AIGVDet)

The base framework introduces a **spatio-temporal anomaly detection pipeline** consisting of:

### 1. Spatial Analysis

* Processes RGB frames using CNN-based feature extraction
* Detects visual artifacts introduced by generative models

### 2. Temporal Analysis

* Uses **RAFT (optical flow)** to model motion between frames
* Identifies unnatural temporal transitions

### 3. Dual-Branch Fusion

* Combines spatial and temporal predictions
* Produces a more robust final classification

### 4. Anomaly Learning Approach

* Focuses on detecting inconsistencies rather than memorizing fake patterns
* Improves generalization across different generation methods

---
## III)Network Architecture

<img width="9600" height="2848" alt="image" src="https://github.com/user-attachments/assets/16b1d4c2-5d1a-45e8-8d0f-188377859ed7" />

---

## IV)Enhancements in AIGVDetPlus

### 1. Performance Optimization

* **Frame Sampling**

  * Reduces redundant computation by processing selected frames
* **Resolution Downscaling**

  * Speeds up inference with minimal accuracy trade-off
* Significantly lowers computational cost compared to the original pipeline

---

### 2. Web-Based Interface (Flask)

* Interactive UI for:

  * Uploading videos
  * Running inference
  * Viewing results
* Removes dependency on command-line execution
* Makes the project demo-friendly and accessible

---

### 3. Automatic Device Selection

* Detects and uses:

  * GPU (if available)
  * CPU (fallback)
* Eliminates manual configuration effort

---

### 4. Structured Scoring Output

* Clean and interpretable prediction results
* Better separation of:

  * Spatial score
  * Temporal score
  * Final fused prediction

---

### 5. Simplified End-to-End Pipeline

* Integrated workflow:

  ```
  Video → Frame Extraction → Optical Flow → Model Inference → Output
  ```
* Reduced setup complexity from the original repository

---

### 6. Improved Code Organization

* Modular and maintainable structure:

  * `checkpoints/` – trained weights
  * `raft_model/` – optical flow model
  * `templates/` – frontend UI
* Clear separation of concerns


---

## V)Performance Comparison (Baseline vs AIGVDetPlus)

| Aspect                 | Original AIGVDet                 | AIGVDetPlus                                     |
| ---------------------- | -------------------------------- | ----------------------------------------------- |
| **Frame Processing**   | Full video frames                | Frame sampling (↓ compute)                      |
| **Resolution**         | Original resolution              | Downscaled frames (↓ memory + speed ↑)          |
| **Pipeline Execution** | Multi-step manual                | Automated end-to-end pipeline                   |
| **Inference Time**     | High (due to full frames + RAFT) | Reduced (sampling + resizing)                   |
| **Hardware Handling**  | Manual device setup              | Automatic CPU/GPU selection                     |
| **Usability**          | CLI-based                        | Flask web interface                             |
| **Output Format**      | Raw predictions                  | Structured scoring (spatial + temporal + fused) |

---

## VI)Estimated Efficiency Gains 

### 1. Frame Sampling

If original processes **N frames**, and you process **k frames**:

* Compute reduction ≈ **N / k**
* Example:

  * 300 frames → 60 frames
  * → **~5× faster preprocessing**

---

### 2. Resolution Downscaling

If resolution reduced from:

* 720p → 360p
* Pixel count ↓ by ~4×

→ CNN + RAFT cost also drops roughly proportionally

---

### 3. Combined Effect

Frame sampling × resolution scaling:

* **Realistic speedup: 3× to 8× faster inference**
* Memory usage: **~50–75% reduction**


## VII)Tech Stack

* **Python 3.10+**
* **PyTorch**
* **Flask**
* **RAFT (Optical Flow)**
* OpenCV, NumPy

---

## VIII)Installation

```bash
git clone https://github.com/Shreyas-Sherekar/AIGVDetPlus
cd AIGVDetPlus
pip install -r requirements.txt
```

---

## IX)Model Weights Setup

Download and place the following:

* `checkpoints/optical.pth`
* `checkpoints/original.pth`
* RAFT weights → `raft_model/`

(Refer to the original repository for download links)

---

## X)Running the Application

For best results use T4 GPU in google colab.

```bash
python app.py
```

Open in browser:

```
http://localhost:5000
```

---



## XI)References

* Original Implementation:
  [https://github.com/multimediaFor/AIGVDet](https://github.com/multimediaFor/AIGVDet)

* Research Paper:
  *AI-Generated Video Detection via Spatio-Temporal Anomaly Learning*

---

## XII)Author

**Shreyas Milind Sherekar**

---

## XIII)Acknowledgment

This project builds upon the original AIGVDet research and aims to make it more accessible, efficient, and deployable.


