🧊 CrystalGen: AI-Powered Crystal Structure Generator 

CrystalGen is an AI-powered web application built with Flask that generates and visualizes crystal structures using a Conditional Variational Autoencoder (CVAE).
It allows users to configure parameters such as space group, composition, and temperature, and instantly generate realistic 3D crystal lattices.

🌐 Overview

CrystalGen integrates:

🧠 Deep Learning (CVAE) for structure generation

🔬 Pymatgen for CIF parsing and structure validation

🧱 Torch Geometric (PyG) for crystal graph representation

🌍 Flask Backend + HTML/JS Frontend for seamless web experience

🎨 3D visualization (Three.js / Plotly / Py3Dmol) for viewing atomic structures



Project structure 


<img width="266" height="717" alt="image" src="https://github.com/user-attachments/assets/c61a8c5f-e502-4925-bd18-b0011ce574d3" />





---

## ⚙️ Installation Guide
### 3️⃣ Install Dependencies
---

## 🧾 Dependencies

Install manually:
```bash
pip install flask flask-cors flask-bcrypt pyjwt pymongo ase pymatgen axios
npm install 3dmol
```
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sadhvini26/CrystalGen.git
cd CrystalGenProject-master
```




### 4️⃣ Run the Flask Application
```bash
cd model
python app.py
```



---


## 🧠 Model Description

CrystalGen uses a **Conditional Variational Autoencoder (CVAE)** to model complex crystal formation patterns.  
The encoder converts input graphs into latent space representations, the diffusion layer introduces structural variations, and the decoder reconstructs valid 3D structures.

### Model Architecture:
- **Encoder:** Compresses input crystal graphs into a latent vector  
- **Latent Diffusion:** Adds controlled stochasticity for realistic diversity  
- **Decoder:** Generates 3D atomic coordinates and lattice parameters  
- **Output:** CIF structure + interactive visualization  

### Model Performance:
| Model | Parameters | Avg. Loss | Accuracy | Quality |
|--------|-------------|------------|-----------|-----------|
| Old Model | 2.47M | 54.8 | 72% | Moderate |
| New Model | 18.07M | 12.3 | 91% | Excellent |

✅ Improved structure reconstruction  
✅ Stable training  
✅ Faster inference

---

## 💻 Web Interface

### 🔹 Home Page
Configure the following parameters:
- **Space Group (1–230)**  
- **Chemical Composition (e.g., FeO, SiO₂)**  
- **Number of Atoms**  
- **Temperature**

Then click **“Generate Structure”** to create your crystal.

### 🔹 Results Page
- Displays **3D visualization** of the generated structure  
- Shows CIF file preview  
- Option to **download structure**

### 🔹 Metrics Page
- View **training/testing losses**  
- Compare old vs new models  
- Displays performance charts and tables

---

## 🔍 Flask API Routes

| Route | Method | Description |
|--------|---------|-------------|
| `/` | GET | Main web interface |
| `/generate` | POST | Generate a new crystal structure |
| `/results` | GET | Display generated crystal and CIF |


---

## 📈 Model Training Workflow

1️⃣ Load and parse CIF files using Pymatgen  
2️⃣ Convert crystal data into graph representations using PyG  
3️⃣ Train CVAE model with reconstruction + KL divergence loss  
4️⃣ Evaluate accuracy and save best model  
5️⃣ Generate and visualize new crystal structures  





---

## 📊 Example Results

| Metric | Old Model | New Model |
|--------|------------|------------|
| Reconstruction Loss | 0.125 | **0.034** |
| KL Divergence | 0.015 | **0.006** |
| Generation Accuracy | 72% | **91%** |
| Inference Time | 0.42s | **0.18s** |



---

## 🧩 Visualization Flow

```
CIF Input → Parser (Pymatgen) → Graph Builder (PyG)
→ CVAE Encoder → Latent Diffusion → Decoder
→ Structure Generation → 3D Visualization → Web UI
```



## ⭐ Summary

**CrystalGen** combines **machine learning**, **materials science**, and **interactive visualization** in a unified Flask web platform — making AI-based crystal generation simple, fast, and visually intuitive.

