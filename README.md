# Industrial 3D Part Segmentation with SAMPart3D

A research project based on the [SAMPart3D](https://github.com/facebookresearch/SAMPart3D) framework, focusing on **multi-granularity zero-shot 3D part segmentation** for **industrial components**, with practical optimizations for robotic grasping tasks.

>  **Project Title:** 基于SAMPart3D框架的工业零件3D部件分割方法研究与优化  
>  **Author:** 卢家骏 | 东南大学人工智能学院  
>  **Duration:** Dec 2024 – May 2025

---

##  Overview

Industrial robotic systems often rely on accurate **3D part-level understanding** to enhance grasp stability and task success rate. This project builds on the SAMPart3D framework to address its limitations when applied to **textureless, complex industrial parts**. We propose three key improvements to adapt the model to real-world industrial use:

-  Color-weak annotations for low-cost structure encoding  
-  Iterative fine-tuning of MLP output layers using a curated training set  
-  Clustering algorithm acceleration for faster inference

---

##  Key Contributions

### 1.  Color-Weak Annotation

Manual weak supervision using color applied to industrial part meshes via Blender.

- Efficient, fast, and intuitive for human operators  
- Boosts segmentation IoU by up to **+32%** in some test cases  
- Requires no semantic labels

### 2.  Iterative MLP Fine-tuning

Few-shot style iterative optimization of output layers.

- Backbone frozen to preserve pre-trained general features  
- Output MLP (instance + position branches) refined across a mini-dataset  
- Improves generalization to new industrial parts

### 3.  Clustering Acceleration

Optimized HDBSCAN clustering during inference:

- Reduces clustering time significantly  
- Preserves segmentation quality  
- Better suited for deployment on industrial hardware

---

##  Dataset

### `Industrial Components-Tiny`

A small-scale curated dataset of **25 industrial part models** with face-level color labels for part boundaries.

- Source: McMaster-CARR CAD repository  
- Format: `.glb` with color-coded surfaces  
- Pipeline: IGS → OBJ → Blender annotation → GLB
- Download:https://huggingface.co/datasets/rubrrr/Industrial_Components-Tiny 
- Example visualization:
![image](https://github.com/user-attachments/assets/21c08837-0f6b-4ef1-a9f9-e71c4587cc86)


---

##  Results

| Method                         | Avg IoU (%) |
|-------------------------------|-------------|
| Original SAMPart3D            | 39.1        |
| + Color-weak annotation       | 51.0        |
| + Iterative fine-tuning (ours)| **58.2**    |

> Test set: 4 industrial parts (俯仰电机安装钣金件, 后罩压线板, 传感器安装座, 偏航摄像机支架)

---

##  Modified files(Compared with original SAMPart3D)

```plaintext
SAMPart3D-industrial/
├── IndustrialComponents-Tiny/           # Training set (25 parts)
│   
├── pointcept/
│   └── engines/
│       └── train.py                     # Compute cluster time
│       └── eval.py                      # Compute cluster time
├── eval/
│   └── eval_part.py                     # Evaluation logic
├── scripts/
│   └── batch_render.sh                  # run /blender_render_16views.py for multiple GLB models
│   └── batch_train.sh                   # Iterative MLP Fine-tuning
├── 2500.pth                             # fine-tuned MLP
└── README.md                            # ← This file
```
## Reference Project

This project is built upon the following open-source work:

**SAMPart3D: Segment Any Part in 3D Objects**  
Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y. Lam, Yan-Pei Cao†, Xihui Liu  
Project Homepage: [https://yhyang-myron.github.io/SAMPart3D-website](https://yhyang-myron.github.io/SAMPart3D-website)




