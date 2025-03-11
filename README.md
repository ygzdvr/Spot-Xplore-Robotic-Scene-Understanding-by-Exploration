# 3D Scene Reconstruction & Semantic Understanding

This repository contains the implementation of a novel approach for **3D scene reconstruction** and **semantic scene understanding** using an **exploratory robotic system**. The goal is to enable a robot to generate an initial low-quality reconstruction of an environment, construct a scene graph, and iteratively refine it by interacting with objects (e.g., opening drawers, doors) to improve semantic understanding.

## 📌 Project Overview

This project builds upon recent advancements in **3D vision and scene representation**, primarily leveraging the concepts introduced in:
- **Spot-Light** ([Paper](https://arxiv.org/abs/2409.11870), [GitHub](http://timengelbracht.github.io/SpotLight/))
- **Spot-Compose** ([Paper](https://arxiv.org/abs/2404.12440), [GitHub](https://spot-compose.github.io/))

### **Key Contributions**
✅ **Incremental 3D Scene Graph Construction**: The robot starts with a coarse scene representation and incrementally refines it through interaction.
✅ **Exploratory Scene Understanding**: The system actively engages with the environment to improve the scene graph, rather than passively observing.
✅ **Semantic Refinement & Object Manipulation**: Objects like doors, drawers, and hidden compartments are detected and manipulated for a more complete understanding.
✅ **Integration of Multi-View Learning**: Incorporating multiple viewpoints to enhance the scene graph and spatial representation.

## 📂 Repository Structure
```
📦 project_root
├── 📂 data                 # Dataset storage
├── 📂 models               # 3D reconstruction and scene understanding models
├── 📂 scripts              # Training, evaluation, and inference scripts
├── 📂 experiments          # Results and logs from different training runs
├── 📂 utils                # Utility functions for data preprocessing, visualization, etc.
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
└── main.py                # Entry point for running experiments
```

## 🚀 Getting Started
### **1. Installation**
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/3D-scene-reconstruction.git
cd 3D-scene-reconstruction
pip install -r requirements.txt
```

### **2. Data Preparation**
Ensure the dataset is available in the `data/` directory. You can use publicly available datasets like **Replica**, **Matterport3D**, or **ScanNet** for testing.

### **3. Training & Inference**
To train the scene reconstruction model:
```bash
python main.py --train --config configs/train_config.yaml
```

To run inference and generate a scene graph:
```bash
python main.py --inference --scene_input data/sample_scene.ply
```

## 🔬 Research Papers & Background
This project is based on prior work in **neural radiance fields (NeRFs)**, **scene graphs**, and **robotic exploration**. For a deeper understanding, refer to:
- [Spot-Light: Neural 3D Scene Understanding](https://arxiv.org/abs/2409.11870)
- [Spot-Compose: Scene Graph Generation for Compositional 3D Scenes](https://arxiv.org/abs/2404.12440)

## 📌 Future Work
🔹 **Active Learning for Scene Refinement**  
🔹 **Integration with SLAM for Robust Mapping**  
🔹 **Better Object Interactivity & Reasoning**  

## ✨ Contributors
- **Your Name** (@yourusername)
- Collaborators

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.

---
🌟 *If you find this project useful, consider giving it a star!* 🌟
