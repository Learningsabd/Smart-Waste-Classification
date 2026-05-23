### Smart Waster Image Classifier
This is a Deep Learning project that classifies waste images into Organic (O) and Recyclable (R) categories performed using two ways : Basic Convolutional Layer and 
Transfer Learning with ResNet50. The goal is to support smarter waste management by automating the sorting process.


## Project Structure

'''Smart Waste Image Classifier/
│
├── DATASET/                  # Training and testing images
│   ├── TRAIN/
│   │   ├── O/                # Organic waste images
│   │   └── R/                # Recyclable waste images
│   └── TEST/
│       ├── O/
│       └── R/
│
├── balanced_waste_images/    # Balanced dataset samples
│
├── main.py                   # Streamlit app for inference
├── Smart_Waste_Image_Classifier.ipynb   # Notebook for experiments
├── Transfer_Learning.ipynb   # Notebook for model training
├── fine-tuned-resnet50.pth   # Saved model weights
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project configuration
├── uv.lock                   # Environment lock file
└── README.md                 # Project documentation'''


## Features
- Comparision between simple CNN and pretrained ResnNet50.
- Two-class classification : Organic Vs Recyclable waste.
- Streamlit Web App for easy image upload and prediction.
- Balanced dataset handling to improve accuracy.


## Getting Started
1. Clone the repository
  git clone https://github.com/Learningsabd/Smart-Waste-Classification.git
  cd Smart-Waste-Classification

2. Install dependencies
   pip install -r requirements.txt

3. Run the Streamlit App
   streamlit run main.py

## Usage
- Upload an image of waste(jpg/jpeg/png).
- The model predicts whether it belongs to Organic or Recyclable.
- Results are displayed instantly in the web app.


  ## Model Details
  - Model : ResNet50 (Unfreeze layer4)
  - Fine-tuned : Final fully connected layer modified for 2 classes.
  - Saved weights : fined-tuned-resnet50.pth
    
