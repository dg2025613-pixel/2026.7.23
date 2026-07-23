import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# 기본 페이지 설정
# ------------------------------
st.set_page_config(page_title="영화 추천 시스템", page_icon="🎬", layout="wide")

st.title("🎬 코사인 유사도 기반 영화 추천 시스템")
st.write("원하는 **장르**와 **세부 특징(분위기, 줄거리 키워드 등)**을 입력하면, "
         "가장 유사한 영화를 리뷰와 함께 추천해드립니다.")


# ------------------------------
# 데이터 로드
# ------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    required_cols = {"title", "genre", "description", "review"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")
    # 결측값 처리
    df = df.fillna("")
    # 장르 + 줄거리를 합쳐서 하나의 텍스트로 (유사도 계산용)
    df["combined_text"] = df["genre"] + " " + df["description"]
    return df


# 사이드바: 자체 CSV 업로드 옵션
with st.sidebar:
    st.header("⚙️ 데이터 설정")
    st.write("기본 샘플 데이터를 사용하거나, 직접 만든 CSV를 업로드할 수 있습니다.")
    st.caption("CSV 필수 컬럼: title, genre, description, review")
    uploaded_file = st.file_uploader("영화 데이터 CSV 업로드", type=["csv"])

    top_n = st.slider("추천 받을 영화 개수", min_value=1, max_value=10, value=5)
    st.divider()
    st.caption("Tip: 장르는 쉼표로 여러 개 입력 가능 (예: 액션, 코미디)")

try:
    if uploaded_file is not None:
        movies_df = load_data(uploaded_file)
    else:
        movies_df = load_data("movies.csv")
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# ------------------------------
# TF-IDF 벡터화 (장르 + 줄거리 기준)
# ------------------------------
@st.cache_resource
def build_vectorizer(corpus: pd.Series):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = build_vectorizer(movies_df["combined_text"])


# ------------------------------
# 사용자 입력
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    genre_input = st.text_input(
        "🎭 장르 입력",
        placeholder="예: 로맨스, 드라마"
    )
with col2:
    detail_input = st.text_area(
        "📝 세부사항 / 분위기 입력",
        placeholder="예: 잔잔하고 감동적인 첫사랑 이야기, 반전이 있는 스릴러 등",
        height=100
    )

search_btn = st.button("🔍 추천 받기", type="primary", use_container_width=True)


# ------------------------------
# 코사인 유사도 계산 & 추천
# ------------------------------
if search_btn:
    if not genre_input.strip() and not detail_input.strip():
        st.warning("장르 또는 세부사항 중 하나는 입력해주세요.")
    else:
        user_text = f"{genre_input} {detail_input}".strip()

        # 사용자 입력을 기존 벡터라이저 어휘 기준으로 변환
        user_vector = vectorizer.transform([user_text])

        # 코사인 유사도 계산 (사용자 입력 vs 전체 영화)
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

        result_df = movies_df.copy()
        result_df["similarity"] = similarities
        result_df = result_df.sort_values(by="similarity", ascending=False)

        top_results = result_df.head(top_n)

        if top_results["similarity"].max() <= 0:
            st.info("입력하신 키워드와 유사한 영화를 찾지 못했습니다. 다른 표현으로 다시 시도해보세요.")
        else:
            st.subheader(f"✨ 추천 영화 Top {top_n}")

            for _, row in top_results.iterrows():
                if row["similarity"] <= 0:
                    continue  # 유사도가 0인 항목은 표시하지 않음

                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### 🎥 {row['title']}")
                        st.markdown(f"**장르:** {row['genre']}")
                    with c2:
                        st.metric("유사도", f"{row['similarity']:.2%}")

                    st.progress(min(float(row["similarity"]), 1.0))

                    st.markdown(f"**📖 줄거리**")
                    st.write(row["description"])

                    st.markdown(f"**💬 리뷰**")
                    st.info(row["review"])

else:
    st.caption("👆 장르와 세부사항을 입력한 뒤 '추천 받기' 버튼을 눌러주세요.")


# ------------------------------
# 전체 영화 데이터 미리보기 (선택)
# ------------------------------
with st.expander("📋 전체 영화 데이터 보기"):
    st.dataframe(
        movies_df[["title", "genre", "description", "review"]],
        use_container_width=True
    )
