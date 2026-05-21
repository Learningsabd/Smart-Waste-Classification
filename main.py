
import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image


# Loading the fine-tuned model

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained = False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('fine-tuned-resnet50.pth', map_location = 'cpu'))
    model.eval()
    return model

model = load_model()


# Defining preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])


# Streamlit UI

st.title('Smart Waste Classification App')
st.write('Upload an image of waste to check for waste classification.')

uploaded_file = st.file_uploader('Choose and image....', type = ['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Show uploaded imahe
    img = Image.open(uploaded_file)
    st.image(img, caption = 'Uploaded Image', use_column_width = True)

    # Preprocess
    input_tensor = transform(img).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim = 1).item()
        
    
    # Map class index to label
    class_names = ['Organic Material', 'Recyclable Material']
    st.success(f'Prediction : ---{class_names[predicted_class]}---')
