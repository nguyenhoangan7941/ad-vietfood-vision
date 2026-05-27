import streamlit as st
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# =========================
# 1. CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="AD VietFood Vision",
    page_icon="🍜",
    layout="wide"
)

# =========================
# 2. CSS GIAO DIỆN
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFE0B2 45%, #FFCCBC 100%);
        color: #3E2723;
    }

    header {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 18px;
        padding-bottom: 35px;
    }

    .logo-box {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0px;
        margin-bottom: 5px;
    }

    .app-logo {
        width: 190px;
        height: 190px;
        object-fit: contain;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.72);
        padding: 10px;
        box-shadow: 0px 8px 24px rgba(230, 81, 0, 0.22);
        border: 2px solid rgba(255, 183, 77, 0.55);
    }

    .brand-title {
        text-align: center;
        font-size: 50px;
        font-weight: 900;
        background: linear-gradient(90deg, #D50000, #FF6D00, #1B5E20);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0px;
        margin-bottom: 3px;
    }

    .brand-slogan {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        color: #1B5E20;
        margin-bottom: 12px;
    }

    .decor-line {
        width: 230px;
        height: 5px;
        background: linear-gradient(90deg, #D50000, #FF6D00, #1B5E20);
        border-radius: 999px;
        margin: 0 auto 22px auto;
    }

    .intro-section {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 24px;
    }

    .intro-title {
        font-size: 30px;
        font-weight: 900;
        color: #E65100;
        margin-bottom: 8px;
    }

    .intro-text {
        font-size: 17px;
        color: #5D4037;
        max-width: 950px;
        margin: 0 auto;
        line-height: 1.7;
    }

    .feature-mini {
        text-align: center;
        padding: 8px 10px;
        margin-bottom: 22px;
    }

    .feature-mini-icon {
        font-size: 36px;
        margin-bottom: 8px;
    }

    .feature-mini-title {
        font-size: 24px;
        font-weight: 900;
        color: #3E2723;
        margin-bottom: 8px;
    }

    .feature-mini-desc {
        font-size: 16px;
        color: #5D4037;
        line-height: 1.5;
    }

    .section-title {
        color: #3E2723;
        font-size: 30px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 18px;
    }

    .food-card {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 12px;
        border-radius: 20px;
        box-shadow: 0px 5px 18px rgba(0, 0, 0, 0.10);
        text-align: center;
        margin-bottom: 15px;
        min-height: 325px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1.5px solid rgba(255, 183, 77, 0.45);
    }

    .food-img {
        width: 100%;
        height: 225px;
        object-fit: cover;
        border-radius: 16px;
        display: block;
    }

    .food-name {
        background-color: #FFF3E0;
        padding: 12px;
        border-radius: 14px;
        text-align: center;
        font-weight: 800;
        color: #E65100;
        border: 2px solid #FFB74D;
        margin-top: 12px;
    }

    .start-note {
        background: linear-gradient(90deg, #FFF3E0, #FFE0B2);
        padding: 16px;
        border-radius: 16px;
        text-align: center;
        font-weight: 800;
        color: #BF360C;
        border: 2px dashed #FB8C00;
        margin-top: 10px;
        margin-bottom: 18px;
        box-shadow: 0px 4px 12px rgba(230, 81, 0, 0.12);
    }

    .predict-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        color: #E65100;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    .predict-subtitle {
        text-align: center;
        font-size: 20px;
        color: #5D4037;
        margin-bottom: 30px;
    }

    .upload-box {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 25px;
        border-radius: 22px;
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.10);
        border: 1.5px solid rgba(255, 183, 77, 0.45);
    }

    .result-box {
        background-color: rgba(255, 255, 255, 0.94);
        padding: 28px;
        border-radius: 22px;
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.12);
        border: 1.5px solid rgba(255, 183, 77, 0.45);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #FF7043, #FFA726);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 14px 34px;
        font-size: 20px;
        font-weight: 800;
        transition: 0.3s;
        box-shadow: 0px 6px 18px rgba(230, 81, 0, 0.25);
    }

    div.stButton > button:hover {
        transform: scale(1.04);
        background: linear-gradient(90deg, #F4511E, #FB8C00);
        color: white;
    }

    .stProgress > div > div > div > div {
        background-color: #FF7043;
    }

    /* KHUNG UPLOAD ẢNH */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #FFFFFF, #FFF3E0);
        padding: 18px;
        border-radius: 20px;
        border: 2px dashed #FF9800;
        box-shadow: 0px 6px 18px rgba(230, 81, 0, 0.15);
    }

    div[data-testid="stFileUploaderDropzone"] {
        background-color: #FFF8E1 !important;
        border: 2px dashed #FFB74D !important;
        border-radius: 16px !important;
        padding: 22px !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #FF7043, #FFA726) !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 900 !important;
    }

    div[data-testid="stFileUploaderDropzone"] button p,
    div[data-testid="stFileUploaderDropzone"] button span,
    div[data-testid="stFileUploaderDropzone"] button div {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }

    div[data-testid="stFileUploaderDropzone"] p,
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] div {
        color: #4E342E !important;
    }

    /* RADIO CHỌN UPLOAD / CAMERA */
    div[role="radiogroup"] label {
        background-color: #FFF3E0 !important;
        padding: 10px 18px !important;
        border-radius: 16px !important;
        border: 1.5px solid #FFB74D !important;
        margin-right: 10px !important;
        color: #4E342E !important;
        font-weight: 800 !important;
    }

    div[role="radiogroup"] label p {
        color: #4E342E !important;
        font-weight: 800 !important;
    }

    /* CAMERA */
    div[data-testid="stCameraInput"] {
        background: linear-gradient(135deg, #FFFFFF, #FFF3E0);
        padding: 18px;
        border-radius: 20px;
        border: 2px dashed #FF9800;
        box-shadow: 0px 6px 18px rgba(230, 81, 0, 0.15);
    }

    /* CHỈNH MÀU CẢNH BÁO */
    div[data-testid="stAlert"] {
        background-color: #FFF3CD !important;
        color: #4E342E !important;
        border-radius: 14px !important;
        border: 1.5px solid #FFB300 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stAlert"] p {
        color: #4E342E !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 3. LOAD MODEL
# =========================

@st.cache_resource
def load_food_model():
    model = load_model("final_foodmodel.h5", compile=False)
    return model

model = load_food_model()

# =========================
# 4. DANH SÁCH CLASS
# =========================
# {'Banh cuon': 0, 'Banh mi': 1, 'Banh xeo': 2, 'Bun bo Hue': 3, 'Com tam': 4}

class_labels = {
    0: "Banh cuon",
    1: "Banh mi",
    2: "Banh xeo",
    3: "Bun bo Hue",
    4: "Com tam"
}

# =========================
# 5. ĐƯỜNG DẪN ẢNH
# =========================

logo_path = "images/logo.png"

food_images = {
    "Banh cuon": "images/banh_cuon.jpg",
    "Banh mi": "images/banh_mi.jpg",
    "Banh xeo": "images/banh_xeo.jpg",
    "Bun bo Hue": "images/bun_bo_hue.jpg",
    "Com tam": "images/com_tam.jpg"
}

# =========================
# 6. SESSION STATE
# =========================

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_predict():
    st.session_state.page = "predict"

def go_to_home():
    st.session_state.page = "home"

# =========================
# 7. HÀM ĐỌC ẢNH BASE64
# =========================

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# =========================
# 8. HÀM TIỀN XỬ LÝ ẢNH
# =========================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((180, 180))
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# =========================
# 9. TRANG CHỦ
# =========================

if st.session_state.page == "home":

    try:
        logo_base64 = get_base64_image(logo_path)

        st.markdown(
            f"""
            <div class="logo-box">
                <img src="data:image/png;base64,{logo_base64}" class="app-logo">
            </div>
            <div class="brand-title">AD VietFood Vision</div>
            <div class="brand-slogan">Nhìn món Việt - Hiểu ẩm thực</div>
            <div class="decor-line"></div>
            """,
            unsafe_allow_html=True
        )

    except:
        st.markdown(
            """
            <div class="brand-title">AD VietFood Vision</div>
            <div class="brand-slogan">Nhìn món Việt - Hiểu ẩm thực</div>
            <div class="decor-line"></div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="intro-section">
            <div class="intro-title">🍽️ Nhận diện món Việt bằng trí tuệ nhân tạo</div>
            <div class="intro-text">
                Ứng dung giúp nhận dạng món ăn Việt Nam từ hình ảnh.
                Chỉ cần tải ảnh lên, hệ thống sẽ phân tích và đưa ra kết quả dự đoán.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3 TÍNH NĂNG CĂN GIỮA

    space_left, feature_col1, feature_col2, feature_col3, space_right = st.columns([0.35, 1, 1, 1, 0.35])

    with feature_col1:
        st.markdown(
            '<div class="feature-mini"><div class="feature-mini-icon">📷</div><div class="feature-mini-title">Tải ảnh</div><div class="feature-mini-desc">Chọn ảnh món ăn từ thiết bị của bạn.</div></div>',
            unsafe_allow_html=True
        )

    with feature_col2:
        st.markdown(
            '<div class="feature-mini"><div class="feature-mini-icon">🤖</div><div class="feature-mini-title">AI dự đoán</div><div class="feature-mini-desc">Mô hình CNN phân tích đặc trưng ảnh.</div></div>',
            unsafe_allow_html=True
        )

    with feature_col3:
        st.markdown(
            '<div class="feature-mini"><div class="feature-mini-icon">📊</div><div class="feature-mini-title">Độ tin cậy</div><div class="feature-mini-desc">Hiển thị xác suất của từng món ăn.</div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            🍽️ Các món ăn hệ thống có thể nhận dạng
        </div>
        """,
        unsafe_allow_html=True
    )

    food_cols = st.columns(5)

    for index, food_name in class_labels.items():
        with food_cols[index]:
            img_path = food_images[food_name]

            try:
                img_base64 = get_base64_image(img_path)

                st.markdown(
                    f"""
                    <div class="food-card">
                        <img src="data:image/jpg;base64,{img_base64}" class="food-img">
                        <div class="food-name">{food_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except:
                st.markdown(
                    f"""
                    <div class="food-card">
                        <div style="
                            height:225px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            background-color:#FFF3E0;
                            border-radius:16px;
                            color:#E65100;
                            font-weight:700;
                        ">
                            Chưa có ảnh minh họa
                        </div>
                        <div class="food-name">{food_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
        """
        <div class="start-note">
            👇 Nhấn nút bên dưới để bắt đầu tải ảnh và nhận dạng món ăn
        </div>
        """,
        unsafe_allow_html=True
    )

    center_col1, center_col2, center_col3 = st.columns([1.4, 1, 1.4])

    with center_col2:
        st.button(
            "🚀 Bắt đầu nhận dạng",
            on_click=go_to_predict,
            use_container_width=True
        )

# =========================
# 10. TRANG CHÍNH NHẬN DẠNG
# =========================

elif st.session_state.page == "predict":

    top_col1, top_col2, top_col3 = st.columns([1, 3, 1])

    with top_col1:
        st.button("⬅ Trang chủ", on_click=go_to_home)

    st.markdown(
        """
        <div class="predict-title">📷 Nhận Dạng Món Ăn</div>
        <div class="predict-subtitle">
            Tải ảnh món ăn lên hoặc chụp trực tiếp để AD VietFood Vision dự đoán tên món ăn.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    input_method = st.radio(
        "Chọn cách đưa ảnh vào hệ thống:",
        ["📁 Tải ảnh từ máy", "📷 Chụp bằng camera"],
        horizontal=True
    )

    uploaded_file = None
    camera_file = None

    if input_method == "📁 Tải ảnh từ máy":
        uploaded_file = st.file_uploader(
            "Chọn ảnh món ăn",
            type=["jpg", "jpeg", "png"]
        )

    elif input_method == "📷 Chụp bằng camera":
        camera_file = st.camera_input("Chụp ảnh món ăn")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    image_source = uploaded_file if uploaded_file is not None else camera_file

    if image_source is not None:
        image = Image.open(image_source)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🖼️ Ảnh đầu vào")
            st.image(
                image,
                caption="Ảnh món ăn",
                use_container_width=True
            )

        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            st.markdown("### ✅ Kết quả dự đoán")

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            food_name = class_labels[predicted_index]

            st.success(f"🍽️ Món ăn dự đoán: **{food_name}**")
            st.info(f"📊 Độ tin cậy: **{confidence:.2f}%**")

            st.markdown("### 📌 Xác suất từng món")

            for i, prob in enumerate(prediction[0]):
                label = class_labels[i]
                percent = prob * 100

                st.write(f"**{label}:** {percent:.2f}%")
                st.progress(float(prob))

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Vui lòng tải ảnh hoặc chụp ảnh món ăn để bắt đầu dự đoán.")
