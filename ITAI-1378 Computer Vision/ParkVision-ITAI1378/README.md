# 🅿️ [ParkVision — Parking Lot Occupancy Detector](https://github.com/YusufShahz/ITAI1378_Midterm_ParkVision) (Click to view seperate repo)

> Real-time parking space detection using YOLOv8 and computer vision.  
> **ITAI 1378 – Computer Vision & AI** · Houston Community College · Yusuf Shahzad

---

## 📌 Problem Statement

30% of urban traffic is caused by drivers circling lots searching for open parking spaces, wasting an average of 8 minutes per driver and costing an estimated $97 billion annually in lost fuel, time, and productivity. Parking lots have no real-time visibility into which spaces are occupied or available — a problem entirely solvable with modern computer vision.

---

## 💡 Solution

ParkVision uses a fine-tuned **YOLOv8s** model to detect and classify individual parking spaces from overhead camera images in real time. Each detected space is labeled as either **occupied** or **empty**, with a visual overlay of colored bounding boxes and a live availability count displayed as a HUD.

**Pipeline:**
```
Overhead Image (640×640 px)
    → Resize + Normalize (RGB)
    → YOLOv8s (73-layer detection model)
    → Classify each bounding box: occupied / empty
    → Output: Visual overlay + availability count
```

---

## 📊 Results

| Metric | Target | Achieved |
|---|---|---|
| mAP@0.5 | ≥ 85% | **99.5%** |
| Precision | — | **99.8%** |
| Recall | — | **99.8%** |
| Inference Speed | < 1000ms | **~17ms (57× faster than target)** |

The model exceeded all performance targets, reaching near-perfect precision and recall after 50 training epochs with early stopping.

---

## 🛠 Technical Approach

| Component | Detail |
|---|---|
| **Model** | YOLOv8s (Ultralytics) — small, fast, pre-trained on COCO |
| **Framework** | PyTorch 2.0 + Ultralytics |
| **CV Task** | Object Detection (bounding box regression + classification) |
| **Input** | 640 × 640 px overhead parking lot images |
| **Output** | Bounding boxes with `occupied` / `empty` class labels + confidence scores |
| **Training** | Google Colab T4 GPU · 50 epochs · early stopping · batch size 16 |

---

## 📁 Dataset

- **Name:** PKLot Dataset
- **Source:** [Roboflow Universe — sagitova-aliya/pklot-qesrf](https://universe.roboflow.com/sagitova-aliya/pklot-qesrf)
- **Size:** ~12,000 labeled overhead parking lot images
- **Classes:** `empty` (available space), `occupied` (space with a vehicle)
- **Split:** 70% Train · 15% Validation · 15% Test

> ⚠️ The dataset is not included in this repository. Download it directly from Roboflow Universe using the link above, or load it via the Roboflow API as shown in the notebook.

---

## ⚙️ Technologies Used

- Python 3.x
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Apache 2.0)
- PyTorch 2.0
- OpenCV
- Google Colab (T4 GPU)
- Roboflow Universe (dataset)

---

## 🚀 How to Run

**Option 1 — Google Colab (Recommended)**

1. Open the notebook: `ParkVision_YOLOv8.ipynb`
2. Click **"Open in Colab"** at the top of the notebook
3. Go to **Runtime → Change runtime type → T4 GPU**
4. Run all cells from top to bottom

**Option 2 — Local**

```bash
pip install ultralytics opencv-python
```

Then open the notebook in Jupyter and run all cells. Ensure you have downloaded the PKLot dataset from Roboflow and updated the dataset path in the config cell.

---

## 📂 Repository Structure

```
ParkVision-ITAI1378/
├── README.md                        # This file
├── ParkVision_YOLOv8.ipynb          # Main Jupyter notebook (all code + outputs)
└── results/
    ├── map_training_curve.png       # mAP@0.5 across 50 epochs
    ├── precision_recall.png         # Precision & recall metrics
    ├── sample_detections/           # Sample output images with bounding boxes
    │   ├── detection_01.jpg
    │   └── detection_02.jpg
    └── model_summary.txt            # YOLOv8s architecture summary
```

---

## 🔑 Key Findings

- **YOLOv8s is highly effective for this task** — even the smallest YOLO variant achieved near-perfect accuracy on the PKLot dataset, thanks to the clear visual distinction between occupied and empty spaces in overhead images.
- **Transfer learning from COCO drastically reduced training time** — the pre-trained weights allowed the model to converge quickly, hitting 97%+ mAP within the first few epochs.
- **Inference speed (17ms) makes real-time deployment practical** — this is fast enough to process live camera feeds at well above 30 FPS, enabling genuine real-world deployment.
- **Early stopping + dropout effectively prevented overfitting** — validation metrics stayed consistent with training metrics throughout, indicating good generalization.

---

## 🧩 Challenges & How They Were Solved

| Challenge | Solution |
|---|---|
| Occluded / overlapping vehicles | Data augmentation; option to scale up to YOLOv8m |
| Dataset mismatch across different parking lots | Supplemented with additional Roboflow datasets |
| Google Colab GPU quota limits | Backup training on Kaggle (30 hrs/week free GPU) |
| Overfitting risk | Early stopping + dropout regularization |

---

## 📫 Contact

**Yusuf Shahzad**  
GitHub: [github.com/YusufShahz](https://github.com/YusufShahz)  
LinkedIn: [linkedin.com/in/yusuf-shahzad-89aba0284](https://www.linkedin.com/in/yusuf-shahzad-89aba0284/)
