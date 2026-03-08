PROJECT STRUCTURE
detector/
│
├── app/
│   ├── model.py            # CNN architecture
│   ├── inference.py        # Model loading and prediction
│   └── preprocess.py       # Image preprocessing
│
├── models/
│   └── synthetic_detector.pth   # Trained model weights
│
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Dependencies
└── README.md

Architecture Overview
Input Image (128x128x3)
        ↓
Conv2D (32 filters)
        ↓
MaxPool
        ↓
Conv2D (64 filters)
        ↓
MaxPool
        ↓
Conv2D (128 filters)
        ↓
MaxPool
        ↓
Flatten
        ↓
Fully Connected Layer
        ↓
Output (Real / Synthetic)

INSTALLATION 

git clone https://github.com/Sornamalya0704/SyntheticImageDetector.git

cd detector

pip install -r requirements.txt

RUNNING THE APPLICATION 

streamlit run frontend.py
