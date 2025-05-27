import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import re

# Cấu hình Streamlit
st.set_page_config(page_title="GEN AI to BI", layout="wide")
st.title("📊 Phân tích dữ liệu CSV bằng AI")

# Upload file CSV
uploaded_file = st.file_uploader("📁 Tải lên file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Dữ liệu mẫu:")
    st.dataframe(df.head())

    # Thống kê
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_info = df.isna().mean().round(3)

    st.markdown("### 📌 Thống kê nhanh")
    st.write(f"- **Số cột numeric:** {len(num_cols)}: {num_cols}")
    st.write(f"- **Số cột category:** {len(cat_cols)}: {cat_cols}")
    st.markdown("**Tỉ lệ thiếu dữ liệu:**")
    st.dataframe(missing_info)

    # Tạo prompt
    sample_data = df.head(3).to_dict()
    prompt = f"""
    Bạn là chuyên gia phân tích dữ liệu. Dưới đây là dataset:
    - Numeric columns: {num_cols}
    - Categorical columns: {cat_cols}
    - Missing data ratio: {missing_info.to_dict()}
    - Sample data: {sample_data}

    Hãy phân tích dataset và:
    1. Đưa ra insight chính.
    2. Gợi ý 1-3 biểu đồ phù hợp với dữ liệu để trực quan hóa.
    Định dạng gợi ý biểu đồ như sau:
    - Chart: [loại biểu đồ: histogram / bar / scatter / boxplot]
    - X: [cột X]
    - Y: [cột Y] (nếu cần)
    """

    st.markdown("### 🤖 Phân tích bằng Mistral (Ollama)")

    if st.button("Phân tích & Vẽ biểu đồ theo LLM"):
        with st.spinner("🔍 Đang phân tích bằng Mistral..."):

            # Gọi API Ollama
            url = "http://localhost:11434/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "mistral",
                "messages": [
                    {"role": "system", "content": "Bạn là nhà phân tích dữ liệu chuyên nghiệp."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }

            try:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                result = response.json()
                content = result['choices'][0]['message']['content']
                st.success("✅ Phân tích hoàn tất!")

                # Hiển thị insight
                st.subheader("📌 Insight từ mô hình:")
                st.write(content)

                # Gợi ý biểu đồ
                st.subheader("📈 Biểu đồ theo gợi ý của LLM:")
                charts = re.findall(r"Chart:\s*(\w+)\s*[\n\r]+X:\s*([\w_]+)(?:\s*[\n\r]+Y:\s*([\w_]+))?", content)

                if not charts:
                    st.warning("🤖 LLM không gợi ý biểu đồ cụ thể nào.")
                else:
                    for i, (chart_type, x_col, y_col) in enumerate(charts):
                        st.markdown(f"**Biểu đồ {i+1}: {chart_type} | X = {x_col}" + (f" | Y = {y_col}" if y_col else "") + "**")
                        fig, ax = plt.subplots()

                        try:
                            if chart_type.lower() == "histogram":
                                sns.histplot(df[x_col], kde=True, ax=ax)
                            elif chart_type.lower() == "bar":
                                sns.barplot(x=df[x_col], y=df[y_col], ax=ax)
                            elif chart_type.lower() == "scatter":
                                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                            elif chart_type.lower() == "boxplot":
                                sns.boxplot(x=df[x_col], y=df[y_col], ax=ax)
                            else:
                                st.warning(f"❌ Không nhận diện được biểu đồ: {chart_type}")
                                continue

                            ax.set_title(f"{chart_type.title()} - {x_col} vs {y_col}" if y_col else f"{chart_type.title()} - {x_col}")
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"🚫 Lỗi khi vẽ biểu đồ {chart_type}: {e}")

            except Exception as e:
                st.error(f"❌ Lỗi khi gọi mô hình Mistral qua Ollama: {e}")
else:
    st.info("📤 Vui lòng tải lên file CSV để bắt đầu.")
