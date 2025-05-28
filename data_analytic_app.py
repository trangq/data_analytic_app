import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI to BI", layout="wide")
st.title("📊 Phân tích dữ liệu CSV")

uploaded_file = st.file_uploader("📁 Tải lên file CSV", type=["csv"])

def draw_chart(df, chart_type, x_cols, y_cols=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type == "bar":
            if y_cols and len(y_cols) == 1:
                # Trường hợp: có cột Y cụ thể
                x = x_cols[0]
                y = y_cols[0]
                sns.barplot(x=df[x], y=df[y], ax=ax, color="#69b3a2")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            else:
                # Trường hợp chỉ có X, không có Y (vẽ value_counts)
                if len(x_cols) == 1:
                    col = x_cols[0]
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, "Số lượng"]
                    sns.barplot(x=col, y="Số lượng", data=counts, ax=ax, color="#69b3a2", ci=None)
                    for i, val in enumerate(counts["Số lượng"]):
                        ax.text(i, val + 0.01 * max(counts["Số lượng"]), str(val), ha="center", va="bottom")
                    ax.set_ylabel("Số lượng")
                    ax.set_xlabel(col)
                else:
                    # Trường hợp nhiều cột X, vẽ stacked/facet-like count
                    all_counts = {}
                    for col in x_cols:
                        counts = df[col].value_counts()
                        all_counts[col] = counts
                    count_df = pd.DataFrame(all_counts).fillna(0).astype(int)
                    count_df.plot(kind="bar", ax=ax, width=0.8, color=sns.color_palette("pastel"))
                    ax.set_ylabel("Số lượng")
                    ax.set_xlabel("Giá trị")
                    ax.legend(title="Cột")

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
                st.warning("Boxplot cần 1 cột X và 1 cột Y.")
                return None
        else:
            st.warning(f"Loại biểu đồ '{chart_type}' chưa hỗ trợ.")
            return None

        ax.set_title(f"{chart_type.title()} - {', '.join(x_cols)}" + (f" vs {', '.join(y_cols)}" if y_cols else ""))
        plt.xticks(rotation=30)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi vẽ biểu đồ: {e}")


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Xem trước dữ liệu:")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("### 📌 Thống kê nhanh")
    st.write(f"- Số cột số (numeric): {len(num_cols)}: {num_cols}")
    st.write(f"- Số cột phân loại (categorical): {len(cat_cols)}: {cat_cols}")
    st.write("**Tỉ lệ thiếu dữ liệu (%):**")
    st.dataframe(df.isnull().mean().round(3))

    # Rule-based Chart Recommendation UI
    st.markdown("---")
    st.markdown("## 📈 Chọn loại biểu đồ ")

    chart_types = ["bar", "histogram", "scatter", "boxplot"]
    selected_chart = st.selectbox("Loại biểu đồ", chart_types)
    selected_x_cols = st.multiselect("Cột X", df.columns.tolist())
    selected_y_cols = st.multiselect("Cột Y (nếu cần)", df.columns.tolist())

    if st.button("🎨 Vẽ biểu đồ"):
        if not selected_x_cols:
            st.warning("Vui lòng chọn ít nhất 1 cột X.")
        else:
            draw_chart(df, selected_chart, selected_x_cols, selected_y_cols if selected_y_cols else None)

else:
    st.info("📥 Vui lòng tải lên file CSV để bắt đầu.")
