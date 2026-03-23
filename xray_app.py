import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# --- 1. 페이지 및 레이아웃 설정 ---
st.set_page_config(page_title="X-Ray 폐렴 분류 + Grad-CAM 시각화", layout="wide")
st.title("🩺 X-Ray 폐렴 분류 + Grad-CAM 시각화")
st.markdown("흉부 X-Ray 이미지를 업로드하면 AI가 폐렴 여부를 진단하고, 판단의 근거가 된 부위를 시각화하여 보여줍니다.")
st.markdown("---")

# --- 2. 모델 로드 (캐싱 적용) ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pneumonia_model.h5')
        return model
    except Exception as e:
        st.error(f"모델 로드 실패: {e}\n'pneumonia_model.h5' 파일이 같은 폴더에 있는지 확인해주세요.")
        return None

model = load_model()

# --- 3. Grad-CAM 함수 (수동 순전파 방식 적용) ---
def get_gradcam_heatmap(img_tensor, model):
    # 마지막 Conv2D 레이어 이름 자동 탐색
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        return np.zeros((150, 150))

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        x = img_tensor
        last_conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_output = x
        preds = x
        class_channel = preds[:, 0] # 이진 분류 기준

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 4. 사이드바 UI 설정 ---
st.sidebar.header("⚙️ 분석 설정")
confidence_threshold = st.sidebar.slider("진단 신뢰도 임계값", min_value=0.3, max_value=0.7, value=0.5, step=0.01)

# --- 5. 메인 앱 로직 ---
uploaded_file = st.file_uploader("흉부 X-Ray 이미지 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # 이미지 읽기 및 전처리
    image = Image.open(uploaded_file).convert('L') # 회색조로 변환
    img_array = np.array(image)
    
    # 150x150 리사이즈 및 정규화
    img_resized = cv2.resize(img_array, (150, 150))
    img_normalized = img_resized / 255.0
    img_tensor = tf.convert_to_tensor(img_normalized.reshape(1, 150, 150, 1), dtype=tf.float32)

    # 예측 수행
    pred_prob = float(model.predict(img_tensor)[0][0])
    is_pneumonia = pred_prob > confidence_threshold
    
    result_text = "폐렴 (Pneumonia)" if is_pneumonia else "정상 (Normal)"
    result_color = "red" if is_pneumonia else "green"

    # Grad-CAM 히트맵 생성 및 오버레이
    heatmap = get_gradcam_heatmap(img_tensor, model)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # 원본 이미지를 RGB로 변환 후 합성
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    # 3열 레이아웃 구성
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("원본 X-Ray")
        st.image(img_array, use_container_width=True, cmap='gray')

    with col2:
        st.subheader("Grad-CAM 분석")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col3:
        st.subheader("AI 예측 결과")
        st.markdown(f"<h3 style='color: {result_color};'>{result_text}</h3>", unsafe_allow_html=True)
        st.metric(label="폐렴 확률", value=f"{pred_prob * 100:.1f}%")
        st.progress(pred_prob, text="AI 확신도 게이지")

    # 결과 이미지 다운로드
    st.markdown("---")
    _, buffer = cv2.imencode(".jpg", overlay)
    st.download_button(
        label="💾 분석 결과 이미지 다운로드",
        data=io.BytesIO(buffer),
        file_name="gradcam_result.jpg",
        mime="image/jpeg"
    )


with open('xray_app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)
print('xray_app.py 저장 완료!')
print('실행: streamlit run xray_app.py')
