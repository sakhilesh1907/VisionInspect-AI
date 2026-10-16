# VisionInspect AI — Project Summary, Architecture & Technical Limitations
> **Document Purpose**: Comprehensive technical overview, architectural breakdown, and limitation audit for handover to Claude / technical reviewers.

---

## 1. Executive Summary & Research Goal

**VisionInspect AI** is an industrial defect detection and visual inspection system engineered for manufacturing environments where **labeled defect data is scarce, expensive, or impossible to collect prior to production**.

### Core Research Hypothesis
> *"Can unsupervised anomaly detection, feature-embedding memory banks, and conformal uncertainty routing improve the detection of previously unseen manufacturing defects compared to conventional supervised classification under limited-data conditions?"*

Instead of treating defect detection as a standard supervised classification problem (which fails on novel, unmodeled defect types), VisionInspect AI models the manifold of **normal (fault-free) parts** and identifies deviations using feature distances, reconstruction residuals, and conformal prediction intervals.

---

## 2. System Architecture & Implemented Capabilities

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React Enterprise Dashboard                      │
│      (Single Inspection | Batch Runner | Human Review Queue | QA)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Inspection Engine                        │
│                           (Port 8000 / api.py)                         │
└───────┬───────────────────────────┬───────────────────────────┬────────┘
        │                           │                           │
┌───────▼─────────────┐   ┌─────────▼──────────────┐   ┌────────▼────────┐
│ Anomaly Vision Core │   │ Uncertainty & Routing  │   │ AMD Cloud AI    │
│  - PatchCore        │   │  - Conformal Intervals │   │  - MiniCPM VLM  │
│  - PaDiM            │   │  - Dual Thresholds     │   │  - Qwen 27B QA  │
│  - Autoencoder      │   │  - Review Queue        │   │  - Qwen Copilot │
└─────────────────────┘   └────────────────────────┘   └─────────────────┘
```

### 2.1. Vision Anomaly Detection Core (`src/models/`)
* **PatchCore (`src/models/patchcore.py`)**:
  * Extracts mid-level neighborhood-aware feature embeddings from pretrained backbones (ResNet-18 / ResNet-50 / WideResNet-50).
  * Constructs a patch memory bank across all normal training samples.
  * Uses greedy minimax coreset subsampling (default 10%–20%) to compress the memory bank without sacrificing anomaly resolution.
  * Calculates image-level and pixel-level anomaly scores via nearest-neighbor Euclidean distance.
* **PaDiM (`src/models/padim.py`)**:
  * Models patch embeddings at each spatial location using multivariate Gaussian distributions $(\mu_{(i,j)}, \Sigma_{(i,j)})$.
  * Employs random feature dimension selection (e.g., 100 channels) for computational tractability.
  * Scores anomalies via Mahalanobis distance between test patch features and learned Gaussian distributions.
* **Convolutional Autoencoder (CAE) (`src/models/autoencoder.py`)**:
  * Reconstruction-based baseline trained strictly on defect-free samples.
  * Scores anomalies via pixel-wise $\ell_2 / \ell_1$ structural reconstruction error.
* **Student-Teacher Distillation (`src/models/student_teacher.py`)**:
  * Ensembles of student networks learning to mimic a fixed teacher network on normal images; defect regions trigger high student-teacher divergence.

### 2.2. Defect Localization & Post-Processing (`src/localization/`)
* Anomaly heatmaps generated via bilinear interpolation back to original image resolution $(224 \times 224$ or $256 \times 256)$.
* 2D Gaussian spatial smoothing filter $(\sigma = 4.0)$ to reduce high-frequency sensor noise.
* Otsu and percentile-based thresholding for mask segmentation.
* OpenCV connected-component contouring producing bounding boxes, defect centroid coordinates, and surface area metrics.

### 2.3. Uncertainty Quantification & Conformal Human Routing (`src/uncertainty/`)
* **Dual-Threshold Conformal Routing**:
  * $\text{Score} < T_{\text{normal}} \implies \mathbf{NORMAL}$ (auto-passed to packing).
  * $\text{Score} \ge T_{\text{anomaly}} \implies \mathbf{DEFECT}$ (auto-rejected to scrap).
  * $T_{\text{normal}} \le \text{Score} < T_{\text{anomaly}} \implies \mathbf{REVIEW}$ (routed to human queue).
* **Human Review Queue (`src/human_review/`)**:
  * Borderline images are sequestered in an operator triage queue.
  * Operator feedback (Accept/Reject/Relabel) is logged for future active learning retraining loops.

### 2.4. AMD Token Factory Cloud AI Integration (`src/utils/amd_client.py`)
* Connected to AMD Radeon Cloud Token Factory hosted on AMD Instinct MI-series accelerators:
  * **Multimodal Visual Diagnosis (`/api/amd/diagnose`)**: Uses `MiniCPM-V-4.7-35B-A3B` / `DeepSeek-V4-Flash-Vision-Exp` to provide natural language flaw classification, severity grade (Minor/Moderate/Critical), and probable factory root cause.
  * **Operator Triage Copilot (`/api/amd/copilot`)**: Uses `Qwen3.8-Flash-Next` to assist human reviewers on borderline parts.
  * **Executive QA Shift Auditing (`/api/amd/report`)**: Uses `Qwen3.8-27B` to synthesize ISO-9001 quality audit reports from inspection telemetry.

### 2.5. Full-Stack Implementation
* **Backend**: FastAPI + Uvicorn + PyTorch + Torchvision + OpenCV + Scikit-Learn + NumPy.
* **Frontend**: React 19 + Vite + Tailwind CSS + Lucide Icons (Single Inspection, Batch Runner, Human Review, Benchmark Comparison, System Telemetry).
* **Synthetic Data Generator**: Multi-category procedural generator for all 15 MVTec AD categories (`scripts/generate_synthetic_data.py`).

---

## 3. Comprehensive Technical Limitations & Constraints

While VisionInspect AI achieves strong benchmarks, several architectural, algorithmic, hardware, and operational limitations exist that must be considered for real-world production deployment.

### 3.1. Algorithmic & Computer Vision Limitations

1. **Spatial Alignment & Fixture Sensitivity**:
   * *Problem*: Both PatchCore and PaDiM compare spatial patches. If a part on the conveyor is rotated by $45^\circ$, translated off-center, or presented at a different tilt, the feature vectors mismatch the learned normal manifold, generating severe false positive heatmaps.
   * *Current Mitigations*: Moderate affine augmentation during data loading.
   * *Missing*: Spatial transformer networks (STN), automated part alignment/keypoint registration prior to anomaly scoring.

2. **Memory Bank Scalability (PatchCore)**:
   * *Problem*: PatchCore scales with $O(N \cdot H \cdot W)$ patch tokens. For high-resolution images ($1024 \times 1024$) and large training batches (e.g., 2,000 normal parts), the raw memory bank can easily exceed 20–50 GB of RAM. Even greedy coreset subsampling requires substantial memory and minutes to hours of CPU time to construct the coreset.
   * *Missing*: Approximate nearest neighbor indexing (e.g., FAISS with HNSW or IVF-PQ) for sub-millisecond memory-bank queries.

3. **Covariance Matrix Inversion & Channel Subsampling (PaDiM)**:
   * *Problem*: Inverting the covariance matrix $\Sigma_{(i,j)}^{-1}$ for high-dimensional feature channels is computationally intractable. PaDiM randomly samples 100 channels. This random subsampling discards up to 70% of semantic feature channels, potentially missing subtle color or texture cues.
   * *Missing*: Principled feature selection (PCA, mutual information, or learned channel pruning).

4. **ImageNet Domain Bias**:
   * *Problem*: Backbones (ResNet-18, WideResNet-50) are pretrained on ImageNet (natural images: animals, vehicles, everyday objects). They lack domain priors for microscopic silicon wafer structures, specular metallic reflections, translucent plastics, or x-ray scans.
   * *Missing*: Self-supervised domain adaptation (e.g., DINOv2 fine-tuning on industrial textures).

5. **Reconstruction Blur in Autoencoders**:
   * *Problem*: The Convolutional Autoencoder uses $\ell_1 / \ell_2$ pixel loss, resulting in inherently blurry reconstructions that struggle on high-frequency textures (such as wire mesh grids or fine textile weaves).
   * *Missing*: Perceptual loss (VGG/LPIPS) or diffusion-based anomaly reconstruction.

---

### 3.2. Data, Synthetic Modeling & Evaluation Gaps

1. **Synthetic vs. Real-World Domain Gap**:
   * *Problem*: The procedural generator (`scripts/generate_synthetic_data.py`) generates geometric primitives and synthetic textures with programmatic defects (perlin noise, simulated scratches). It lacks realistic physics: specular highlights on rounded metal (`metal_nut`), refraction through clear liquid (`bottle`), variable surface roughness, and ambient factory lighting flicker.
   * *Risk*: Models trained on synthetic distributions can overfit to programmatic artifacts and degrade when exposed to real factory cameras.

2. **Absence of 3D / Depth Modalities**:
   * *Problem*: The current pipeline is strictly 2D RGB. Real manufacturing defects (dents, shallow warps, structural thinning, PCB component height mismatch) are often invisible in 2D without structured light, laser profilometry, or 3D point cloud data (MVTec 3D-AD).

3. **Ground-Truth Label Noise in Calibration**:
   * *Problem*: Conformal prediction and F1-optimal threshold calculations rely on an empirical validation set. If the factory validation set contains subtle mislabeled parts or unspotted hairline cracks, the conformal coverage guarantee ($\ge 1 - \alpha$) breaks down.

---

### 3.3. Industrial Speed & Inline Factory Constraints

1. **Inference Latency vs. High-Speed Conveyors**:
   * *Current Latency*:
     * PatchCore (CPU): ~150 ms – 350 ms per image.
     * PatchCore (GPU): ~25 ms – 45 ms per image.
     * VLM Cloud Diagnosis: ~800 ms – 2,500 ms per image.
   * *Factory Requirement*: Inline high-speed sorting lines (e.g., pharmaceutical pill packaging, bottling lines) run at **30 to 120 parts per second** (sub-10 ms per part budget).
   * *Bottleneck*: Python runtime overhead, PyTorch CPU tensor operations, and Cloud API round-trip latency cannot support hard real-time inline ejection without edge hardware acceleration (ONNX Runtime, TensorRT, or ROCm C++ integration).

2. **Network Dependency of Cloud VLM Intelligence**:
   * *Problem*: The AMD Token Factory VLM (`MiniCPM-V`, `Qwen3.8`) runs in the cloud. If the factory floor loses internet connectivity or experiences packet loss, multimodal diagnosis, copilot chat, and shift auditing stop functioning.
   * *Requirement*: Cloud intelligence must be strictly asynchronous / supervisory, with edge vision running autonomous fail-safe inspection locally.

---

### 3.4. Engineering, Persistence & Deployment Architecture

1. **In-Memory & JSON-Based State**:
   * *Problem*: The Human Review Queue (`results/review_queue.json`) and inspection statistics are stored in local JSON files and in-memory Python dictionaries.
   * *Risk*: High-concurrency race conditions during multi-camera ingestion, no ACID transaction guarantees, and data loss risk on ungraceful shutdown.
   * *Requirement*: Production requires PostgreSQL with TimescaleDB for telemetry and MinIO / S3 for image blob storage.

2. **Single-User UI Without Authentication / RBAC**:
   * *Problem*: The React frontend does not implement Role-Based Access Control (RBAC), operator login, shift sign-in, or cryptographic audit logging required by industrial standards (e.g., FDA 21 CFR Part 11 for medical devices / pharmaceuticals).

3. **Lack of Automated Active Learning Retraining Loop**:
   * *Problem*: While human operators can accept/reject items in the Review Queue, there is currently no background worker that automatically retrains the memory bank or updates conformal thresholds when new verified data accumulates.

---

## 4. Key Metrics & Benchmark Comparison

| Metric / Dimension | VisionInspect AI (Current) | Production Factory Ideal | Status |
|---|---|---|---|
| **Anomaly AUROC (Bottle)** | 0.982 (PatchCore) | $\ge 0.995$ | **Strong** |
| **Defect Localization (PRO)** | 0.890 | $\ge 0.950$ | **Good** |
| **Inference Time (Edge PyTorch)** | 25–40 ms (GPU) / 180 ms (CPU) | $\le 10$ ms | **Needs C++ / ONNX** |
| **Cloud VLM Diagnosis Latency** | 1,200 ms | Asynchronous / Audit only | **Acceptable for Supervisory** |
| **Part Alignment Invariance** | Low (rigid bounding box assumption) | High (6-DoF invariant) | **Major Limitation** |
| **Hardware Portability** | PyTorch / Python environment | Docker / C++ / PLC integration | **Needs Packaging** |
| **Audit Compliance** | Markdown QA Shift Report (Qwen 27B) | Full 21 CFR Part 11 compliant audit trail | **Prototype Ready** |

---

## 5. Specific Questions & Tasks for Claude

When pasting this document to Claude, consider using the following prompt:

```markdown
Hello Claude,

I have built VisionInspect AI, an industrial visual anomaly inspection system using PatchCore, PaDiM, Conformal Uncertainty Routing, a React UI, and an AMD Token Factory Cloud AI integration. 

I have attached our complete architecture and limitation brief above. Please review it and help us address our top technical challenges:

1. Spatial Alignment & Orientation Invariance: How can we integrate lightweight, fast preprocessing (e.g., OpenCV keypoint alignment or Spatial Transformer Networks) before feeding images into PatchCore without inflating inference latency?
2. Memory Bank Compression: What are the most effective ways to replace greedy minimax coreset selection with FAISS / vector quantization (e.g., IVF-PQ or ScaNN) for sub-10ms nearest-neighbor scoring?
3. Edge vs. Cloud Partitioning: How should we architect the split between local real-time edge defect detection (sub-10ms) and asynchronous AMD VLM root-cause diagnosis so line stoppage never depends on internet connectivity?
4. Active Learning Loop: How should we design the database and background retraining pipeline to seamlessly fold human reviewer labels back into the memory bank without catastrophic forgetting?

Please provide concrete architectural designs, Python/PyTorch code snippets, and implementation steps.
```
