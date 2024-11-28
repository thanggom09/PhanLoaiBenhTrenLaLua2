import torch
import os
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from torchvision import transforms

# --- Cài đặt giao diện ---
st.set_page_config(page_title="Phân Loại & Tư Vấn Bệnh Lá Lúa", layout="wide", page_icon="🌾")

# --- Tải mô hình phân loại bệnh lá ---
@st.cache_resource
def load_disease_model(model_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.last_channel, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 8),
        torch.nn.Softmax(dim=1),
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model_path = 'model/mobilenetv2_trained_model.pth'  # Đường dẫn mô hình phân loại
if os.path.exists(model_path):
    disease_model = load_disease_model(model_path)
else:
    st.error("Không tìm thấy mô hình phân loại!")

# --- Các nhãn bệnh ---
disease_labels = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# --- Dữ liệu các biện pháp khắc phục bệnh ---
disease_remedies = {
    "Bacterial Leaf Blight": "Sử dụng các giống kháng bệnh và phun thuốc gốc đồng như Copper Oxychloride.",
    "Brown Spot": "Phun thuốc trừ bệnh chứa Mancozeb hoặc Carbendazim. Bón phân cân đối.",
    "Healthy Rice Leaf": "Không cần xử lý. Duy trì chăm sóc tốt để phòng bệnh.",
    "Leaf Blast": "Sử dụng thuốc trừ bệnh như Tricyclazole. Bón phân cân đối để tăng sức đề kháng.",
    "Leaf Scald": "Giảm lượng phân đạm và sử dụng thuốc bảo vệ thực vật phù hợp.",
    "Narrow Brown Leaf Spot": "Phun Mancozeb hoặc Zineb. Tránh tưới nước quá mức.",
    "Rice Hispa": "Dùng thuốc trừ sâu chứa Chlorpyrifos hoặc Quinalphos.",
    "Sheath Blight": "Phun thuốc Validamycin hoặc Hexaconazole. Duy trì mật độ gieo trồng hợp lý.",
}

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    return image

# --- Hàm kiểm tra lá lúa ---
def is_rice_leaf(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255, 255))
    green_ratio = cv2.countNonZero(green_mask) / (image.shape[0] * image.shape[1])
    return green_ratio > 0.5

# --- Giao diện chính ---
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Phân Loại & Tư Vấn Bệnh Lá Lúa 🌾</h1>", unsafe_allow_html=True)

# --- Tải lên ảnh ---
uploaded_image = st.file_uploader("Tải lên ảnh lá lúa:", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    if is_rice_leaf(image):
        st.markdown("<h3 style='color: green;'>✅ Đây là lá lúa, đang phân loại bệnh...</h3>", unsafe_allow_html=True)

        # Tiền xử lý ảnh và dự đoán
        processed_image = preprocess_image(image)
        with torch.no_grad():
            prediction = disease_model(processed_image)[0]
        max_probability = torch.max(prediction).item()
        predicted_label = disease_labels[torch.argmax(prediction).item()]

        # Hiển thị kết quả phân loại
        if max_probability >= 0.5:
            st.markdown("<h3 style='color: green;'>🌟 Kết quả phân loại:</h3>", unsafe_allow_html=True)
            st.success(f"{predicted_label}: {max_probability * 100:.2f}%")

            # Hiển thị biện pháp khắc phục từ dictionary
            remedy = disease_remedies.get(predicted_label, "Không có thông tin về biện pháp khắc phục bệnh này.")
            st.markdown("<h3 style='color: blue;'>💡 Biện pháp khắc phục:</h3>", unsafe_allow_html=True)
            st.success(remedy)

        else:
            st.warning("Không thể xác định bệnh với độ chính xác cao.")
    else:
        st.markdown("<h3 style='color: red;'>⚠️ Đây không phải là lá lúa.</h3>", unsafe_allow_html=True)
