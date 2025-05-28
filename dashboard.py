import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re

# Cấu hình Streamlit
st.set_page_config(page_title="GEN AI to BI", layout="wide")
st.title("📊 Phân tích dữ liệu CSV bằng AI (Tiếng Việt)")

# Upload file CSV
uploaded_file = st.file_uploader("📁 Tải lên file CSV", type=["csv"])

def draw_chart(df, chart_type, x_cols, y_cols=None):
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        if chart_type == "bar":
            if y_cols and len(y_cols) == 1:
                y_col = y_cols[0]
                # Nếu y_col là "Số lượng khách hàng", đếm số lượng theo x_col
                if y_col.lower() in ['số lượng khách hàng', 'solượngkháchhàng', 'soluongkhachhang']:
                    counts = df[x_cols[0]].value_counts().reset_index()
                    counts.columns = [x_cols[0], y_col]
                    sns.barplot(x=x_cols[0], y=y_col, data=counts, ax=ax, edgecolor='black', linewidth=0)
                    ax.set_xlabel(x_cols[0])
                    ax.set_ylabel(y_col)
                    # Thêm label số lượng trên bar
                    for i, val in enumerate(counts[y_col]):
                        ax.text(i, val + max(counts[y_col])*0.01, str(val), ha='center', va='bottom')
                else:
                    sns.barplot(x=df[x_cols[0]], y=df[y_col], ax=ax, edgecolor='black', linewidth=0)
                    ax.set_xlabel(x_cols[0])
                    ax.set_ylabel(y_col)
            else:
                counts = df[x_cols[0]].value_counts()
                sns.barplot(x=counts.index, y=counts.values, ax=ax, edgecolor='black', linewidth=0)
                ax.set_xlabel(x_cols[0])
                ax.set_ylabel("Số lượng")
                for i, val in enumerate(counts.values):
                    ax.text(i, val + max(counts.values)*0.01, str(val), ha='center', va='bottom')

        elif chart_type == "histogram":
            for col in x_cols:
                sns.histplot(df[col].dropna(), kde=True, ax=ax, label=col, alpha=0.5)
            ax.legend()
        elif chart_type == "scatter":
            if len(x_cols) >= 1 and y_cols and len(y_cols) >= 1:
                sns.scatterplot(x=df[x_cols[0]], y=df[y_cols[0]], ax=ax)
            else:
                st.warning("Scatter cần 1 cột X và 1 cột Y.")
                return None
        elif chart_type == "boxplot":
            if len(x_cols) >= 1 and y_cols and len(y_cols) == 1:
                sns.boxplot(x=df[x_cols[0]], y=df[y_cols[0]], ax=ax)
            else:
                st.warning("Boxplot cần 1 cột category (X) và 1 cột numeric (Y).")
                return None
        else:
            st.warning(f"Loại biểu đồ '{chart_type}' chưa hỗ trợ.")
            return None

        ax.set_title(f"{chart_type.title()} - {', '.join(x_cols)}" + (f" vs {', '.join(y_cols)}" if y_cols else ""))
        plt.xticks(rotation=30)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ: {e}")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Dữ liệu mẫu:")
    st.dataframe(df.head())

    # Thống kê cột numeric và categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_info = df.isna().mean().round(3)

    st.markdown("### 📌 Thống kê nhanh")
    st.write(f"- **Số cột numeric:** {len(num_cols)}: {num_cols}")
    st.write(f"- **Số cột category:** {len(cat_cols)}: {cat_cols}")
    st.markdown("**Tỉ lệ thiếu dữ liệu:**")
    st.dataframe(missing_info)

    # Tạo prompt bằng tiếng Việt, yêu cầu định dạng trả lời rõ ràng
    sample_data = df.head(3).to_dict()
    prompt = f"""
Bạn là chuyên gia phân tích dữ liệu chuyên nghiệp.
Dataset có các thông tin sau:
- Các cột số: {num_cols}
- Các cột phân loại: {cat_cols}
- Tỉ lệ dữ liệu thiếu: {missing_info.to_dict()}
- Ví dụ dữ liệu mẫu: {sample_data}

Hãy:
1. Đưa ra các insight chính bằng tiếng Việt.
2. Gợi ý 1 đến 3 biểu đồ phù hợp để trực quan hóa dữ liệu.
3. Định dạng gợi ý biểu đồ theo mẫu sau (bắt buộc chính xác):
- Chart: [histogram / bar / scatter / boxplot]
- X: [tên cột]
- Y: [tên cột] (có thể không có nếu không cần)
Nếu không có Y thì bỏ dòng Y.
"""

    st.markdown("### 🤖 Phân tích & gợi ý biểu đồ bằng Mistral (Ollama)")

    if st.button("Phân tích & Vẽ biểu đồ theo LLM"):
        with st.spinner("Đang gọi Mistral phân tích..."):
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
                response = requests.post(url, headers=headers, json=data)
                result = response.json()

                # Lấy content trả về
                if 'message' in result and 'content' in result['message']:
                    content = result['message']['content']
                elif 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                else:
                    st.error("Mô hình không trả về nội dung hợp lệ.")
                    st.stop()

                # Hiển thị insight
                st.subheader("📌 Insight từ mô hình (tiếng Việt):")
                st.write(content)

                # Regex lấy gợi ý biểu đồ
                charts = re.findall(
                    r"Chart:\s*(\w+).*?X:\s*([\w_]+)(?:.*?Y:\s*([\w_]+))?",
                    content, flags=re.IGNORECASE | re.DOTALL)

                if not charts:
                    st.warning("🤖 LLM không gợi ý biểu đồ cụ thể nào.")
                else:
                    st.subheader("📈 Biểu đồ theo gợi ý của LLM:")
                    # Vẽ từng biểu đồ
                    for idx, (chart_type, x_col, y_col) in enumerate(charts):
                        st.markdown(f"**Biểu đồ {idx+1}:** {chart_type.title()} | X: {x_col}" + (f" | Y: {y_col}" if y_col else ""))
                        draw_chart(df, chart_type.lower(), [x_col], [y_col] if y_col else None)

            except Exception as e:
                st.error(f"Lỗi khi gọi mô hình Mistral qua Ollama: {e}")

    # --- Phần cho phép người dùng chọn thủ công ---
    st.markdown("---")
    st.markdown("## 🛠️ Vẽ biểu đồ tùy chọn")

    chart_types = ["histogram", "bar", "scatter", "boxplot"]
    selected_chart = st.selectbox("Chọn loại biểu đồ", chart_types)

    # Cho phép chọn nhiều cột (danh sách)
    selected_x_cols = st.multiselect("Chọn cột X (có thể chọn nhiều)", df.columns.tolist())
    selected_y_cols = st.multiselect("Chọn cột Y (có thể chọn nhiều)", df.columns.tolist())

    if st.button("Vẽ biểu đồ tùy chọn"):
        if not selected_x_cols:
            st.warning("Vui lòng chọn ít nhất 1 cột X.")
        else:
            draw_chart(df, selected_chart, selected_x_cols, selected_y_cols if selected_y_cols else None)

else:
    st.info("📤 Vui lòng tải lên file CSV để bắt đầu.")
