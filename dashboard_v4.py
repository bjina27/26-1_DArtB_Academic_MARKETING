# dashboard.py
# 실행: python -m streamlit run C:\Users\SAMSUNG\Desktop\ohouse_final\dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import base64
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / "ohou.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="오늘의집 맞춤 상품 추천",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════
# 멀티페이지 네비게이션
# ══════════════════════════════════════════
dashboard_page = st.Page("pages/0_dashboard_main.py", title="대시보드", icon="🏠", default=True)
test_page = st.Page("pages/1_공간인식유형테스트.py", title="공간인식유형 테스트", icon="🧪")
nav = st.navigation([dashboard_page, test_page], position="hidden")
nav.run()
st.stop()

# 아래 코드는 실행되지 않음 (pages/0_dashboard_main.py 로 이동)
# ══════════════════════════════════════════
# session_state 초기화
# ══════════════════════════════════════════
for key, default in [
    ("candidate_ids", None),
    ("filtered_len", None),
    ("recommend_done", False),
    ("change_type_saved", None),
    ("space_saved", None),
    ("selected_cats_saved", None),
    ("depth1_saved", None),
    ("depth2_saved", None),
    ("depth3_saved", None),
    ("synth_result", None),
    ("synth_product", None),
    ("room_img_bytes", None),
    ("selected_combo_idx", None),
    ("selected_products", {}),  # 카테고리별 선택된 상품 저장
    ("show_space_test", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════
@st.cache_resource
def load_model():
    with open(BASE / "recommend_model_final.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_category_tree():
    return pd.read_csv(
        BASE / "housewarming_product_tags_with_image.csv",
        usecols=["place_label", "ohou_category_depth1",
                 "ohou_category_depth2", "ohou_category_depth3"],
        low_memory=False
    ).dropna(subset=["ohou_category_depth1"])

with st.spinner("모델 로딩 중..."):
    model        = load_model()
    posts        = model["posts"]
    tags_valid   = model["tags_valid"]
    scaler       = model["scaler"]
    feature_cols = model["feature_cols"]
    VALID_PLACES = model["VALID_PLACES"]
    KNN_CAT_COLS = model["KNN_CAT_COLS"]
    KNN_NUM_COLS = model["KNN_NUM_COLS"]
    K_DEFAULT    = model.get("K_DEFAULT", 20)
    K_FALLBACK   = model.get("K_FALLBACK", 50)
    MIN_RESULT   = model.get("MIN_RESULT", 5)
    MIN_COUNT    = model.get("MIN_COUNT", 3)
    FREQ_W       = model.get("FREQ_W", 0.4)
    POP_W        = model.get("POP_W", 0.6)
    grid_representative = model["grid_representative"]
    tags_cat     = load_category_tree()

SPACE_TYPE_OPTIONS = {
    "": "선택 안 함",
    "TR": "TR — 감성 우선형",
    "RT": "RT — 편안함 우선형",
    "TH": "TH — 취향 작업실형",
    "TF": "TF — 공유 감성형",
    "HT": "HT — 아지트 감성형",
    "RH": "RH — 나만의 루틴형",
    "HR": "HR — 활동적 휴식형",
    "RF": "RF — 포근한 보금자리형",
    "TT": "TT — 극강 취향형",
    "FR": "FR — 따뜻한 함께형",
    "FT": "FT — 감성 모임형",
    "RR": "RR — 충전소형",
    "FH": "FH — 활동 공유형",
    "HF": "HF — 함께 활동형",
    "HH": "HH — 극강 작업실형",
    "FF": "FF — 공동체형",
}


def inject_dashboard_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap');
        html, body, [class*="css"]  {font-family:'Noto Sans KR', sans-serif;}
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {display:none !important;}
        .stApp {background:#ffffff; color:#2f3438;}
        .block-container {padding-top: 2.5rem !important; padding-bottom: 4rem; max-width: 1380px; overflow: visible !important;}
        .oh-header{
          display:flex; align-items:center; justify-content:flex-start;
          padding:.6rem 0 1rem; border-bottom:1px solid #eaedef; margin-bottom:1rem;
          overflow: visible;
        }
        .oh-brand{display:flex; align-items:center; gap:1rem; overflow: visible;}
        .oh-logo{display:flex; align-items:center; gap:.75rem; font-size:1.75rem; font-weight:800; color:#2f3438; line-height:1.3; overflow: visible;}
        .oh-badge{
          width:46px; height:46px; border-radius:0; background:transparent;
          display:flex; align-items:center; justify-content:center; flex-shrink:0;
        }
        .oh-subnav-wrap{display:none;}
        .hero-wrap{
          padding: 0;
          border: 1px solid #eaedef;
          border-radius: 28px;
          background: #fff;
          box-shadow: 0 10px 30px rgba(47,52,56,.05);
          overflow:hidden;
          margin-bottom: 1.1rem;
        }
        .hero-slider{
          position:relative; min-height:430px; overflow:hidden; border-radius:28px;
        }
        .hero-track{
          display:flex; width:300%; height:430px; animation:heroSlide 18s infinite;
        }
        .hero-slide{
          width:33.3333%; height:430px; position:relative; flex-shrink:0;
          background-size:cover; background-position:center;
        }
        .hero-slide::after{
          content:''; position:absolute; inset:0;
          background:linear-gradient(0deg, rgba(20,24,28,.58) 0%, rgba(20,24,28,.18) 58%);
        }
        .hero-content{
          position:absolute; left:2rem; right:2rem; bottom:2rem; z-index:2; color:#fff;
        }
        @keyframes heroSlide{
          0%, 28% {transform:translateX(0);}
          33%, 61% {transform:translateX(-33.3333%);}
          66%, 94% {transform:translateX(-66.6666%);}
          100% {transform:translateX(0);}
        }
        .hero-kicker{
          display:inline-block;
          width:fit-content;
          padding: .45rem .85rem;
          border-radius: 999px;
          background: rgba(53,197,240,.92);
          color: #fff;
          font-size: .8rem;
          font-weight: 700;
          margin-bottom: .9rem;
        }
        .hero-title{
          font-size: 3.1rem;
          line-height: 1.2;
          font-weight: 800;
          color: #fff;
          letter-spacing: -.03em;
          margin: 0 0 .55rem 0;
        }
        .hero-copy{
          color:rgba(255,255,255,.88);
          font-size: 1.02rem;
          line-height: 1.75;
          margin: 0;
        }
        .stat-panel{
          height:100%;
          padding:1.5rem;
          background:#4b63f3;
          color:#fff;
          border-radius:28px;
          box-shadow:0 10px 30px rgba(75,99,243,.18);
        }
        .stat-panel h3{
          margin:0 0 1rem 0; font-size:1.7rem; font-weight:800;
        }
        .hero-stat{
          background: transparent;
          border-bottom: 1px solid rgba(255,255,255,.18);
          border-radius: 0;
          padding: 1.1rem 0;
          text-align:left;
          min-height: unset;
        }
        .hero-stat-num{
          display:block;
          color:#fff;
          font-size:2rem;
          font-weight:800;
          margin-top:.2rem;
          text-align:right;
        }
        .hero-stat-lbl{
          color:rgba(255,255,255,.86);
          font-size:.98rem;
          font-weight:600;
        }
        .cta-strip{
          margin-top:1rem; padding:1.6rem 1.8rem; border-radius:24px;
          background:linear-gradient(90deg, #2e3c98 0%, #4b63f3 100%); color:#fff;
          display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
        }
        .cta-title{font-size:1.7rem; font-weight:800; margin:0 0 .25rem 0;}
        .cta-copy{font-size:1rem; color:rgba(255,255,255,.84); margin:0;}
        .cta-btn{
          display:inline-flex; align-items:center; gap:.6rem;
          padding:1rem 2rem; border-radius:16px;
          background:#fff; color:#4b63f3;
          font-size:1.05rem; font-weight:800;
          text-decoration:none; white-space:nowrap; flex-shrink:0;
          transition: opacity .15s;
        }
        .cta-btn:hover{opacity:.88; color:#4b63f3;}
        /* 테스트 시작 page_link 스타일 */
        [data-testid="stPageLink-NavLink"] {
          background:#fff !important;
          color:#4b63f3 !important;
          border-radius:16px !important;
          padding:.85rem 1.5rem !important;
          font-weight:800 !important;
          font-size:1.05rem !important;
          text-align:center !important;
          border:none !important;
          box-shadow:0 4px 12px rgba(0,0,0,.08) !important;
        }
        [data-testid="stPageLink-NavLink"]:hover {
          opacity:.88 !important;
        }
        .filter-wrap{
          padding: 1rem 0 1.1rem;
          border-bottom: 1px solid #eaedef;
          border-radius: 0;
          background: #ffffff;
          box-shadow: none;
          margin-bottom: 1rem;
        }
        .filter-title{
          font-size: 1.05rem;
          font-weight: 800;
          color: #2f3438;
          margin-bottom: .9rem;
        }
        .filter-meta{display:flex; justify-content:space-between; align-items:center; margin:.4rem 0 .8rem;}
        .filter-meta span{color:#7b858e; font-weight:600;}
        .stSelectbox label, .stSlider label, .stFileUploader label {
          display:block !important;
          font-size:.95rem !important;
          font-weight:700 !important;
          color:#2f3438 !important;
          margin-bottom:.35rem !important;
        }
        /* placeholder 회색 처리 */
        [data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] span,
        [data-baseweb="select"] .css-placeholder,
        [data-baseweb="select"] [class*="placeholder"] {
          color: #aab0b7 !important;
        }
        /* 선택 안된 selectbox 내부 텍스트 회색 */
        [data-baseweb="select"] > div > div[aria-selected="false"] span {
          color: #aab0b7 !important;
        }
        .section-card{
          padding: 1rem 1.1rem;
          border: 1px solid #eaedef;
          border-radius: 20px;
          background: #ffffff;
          margin-bottom: 1rem;
        }
        .guide-title{font-size:2rem; font-weight:800; margin-bottom:.3rem; color:#2f3438;}
        .guide-copy{font-size:1rem; color:#7b858e; margin-bottom:1.1rem;}
        @media (max-width: 1100px){
          .oh-header, .oh-brand, .filter-meta, .cta-strip {display:block;}
          .hero-title{font-size:2.2rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════
# 추천 유틸 함수
# ══════════════════════════════════════════
def build_feature_matrix(df):
    df = df.copy()
    for col in KNN_CAT_COLS:
        df[col] = df[col].fillna("unknown").astype(str)
    encoded = pd.get_dummies(df[KNN_CAT_COLS], prefix=KNN_CAT_COLS)
    num     = df[KNN_NUM_COLS].fillna(0)
    return pd.concat([
        df[["contentId"]].reset_index(drop=True),
        encoded.reset_index(drop=True),
        num.reset_index(drop=True)
    ], axis=1)


def get_candidates(filtered, user_row):
    X_filtered = build_feature_matrix(filtered)
    for col in feature_cols:
        if col not in X_filtered.columns:
            X_filtered[col] = 0
    X_filtered_scaled = scaler.transform(X_filtered[feature_cols].values)

    X_user = build_feature_matrix(user_row)
    for col in feature_cols:
        if col not in X_user.columns:
            X_user[col] = 0
    user_scaled = scaler.transform(X_user[feature_cols].values)

    k_actual = min(K_DEFAULT, len(filtered))
    knn = NearestNeighbors(n_neighbors=k_actual, metric="cosine",
                           algorithm="brute", n_jobs=-1)
    knn.fit(X_filtered_scaled)
    _, indices = knn.kneighbors(user_scaled)
    candidate_ids = filtered.iloc[indices[0]]["contentId"].values

    check = tags_valid[tags_valid["contentId"].isin(candidate_ids)]
    if len(check) < MIN_RESULT:
        k_actual = min(K_FALLBACK, len(filtered))
        knn2 = NearestNeighbors(n_neighbors=k_actual, metric="cosine",
                                algorithm="brute", n_jobs=-1)
        knn2.fit(X_filtered_scaled)
        _, indices = knn2.kneighbors(user_scaled)
        candidate_ids = filtered.iloc[indices[0]]["contentId"].values

    return candidate_ids


def recommend_by_category(candidate_ids, space, depth2, top_n=4):
    candidate_tags = tags_valid[tags_valid["contentId"].isin(candidate_ids)].copy()
    if space:
        candidate_tags = candidate_tags[candidate_tags["place_label"] == space]

    used_level = depth2
    if depth2:
        sub = candidate_tags[candidate_tags["ohou_category_depth2"] == depth2]
        if len(sub) >= MIN_COUNT:
            candidate_tags = sub
        elif len(sub) > 0:
            candidate_tags = sub
        else:
            return pd.DataFrame(), used_level

    candidate_tags = candidate_tags.merge(
        posts[["contentId", "popularity_score"]], on="contentId", how="left"
    )
    product_score = (
        candidate_tags.groupby([
            "productId", "productName", "brand", "sellingPrice",
            "ohou_category_depth1", "ohou_category_depth2",
            "originalImageUrl", "productUrl"
        ])
        .agg(등장수=("contentId","count"), 평균인기도=("popularity_score","mean"))
        .reset_index()
    )
    product_score = product_score[product_score["등장수"] >= 2]
    if len(product_score) == 0:
        product_score = (
            candidate_tags.groupby([
                "productId", "productName", "brand", "sellingPrice",
                "ohou_category_depth1", "ohou_category_depth2",
                "originalImageUrl", "productUrl"
            ])
            .agg(등장수=("contentId","count"), 평균인기도=("popularity_score","mean"))
            .reset_index()
        )

    if len(product_score) == 0:
        return pd.DataFrame(), used_level

    max_log = np.log1p(product_score["등장수"]).max()
    max_pop = product_score["평균인기도"].max()
    product_score["최종점수"] = (
        np.log1p(product_score["등장수"]) / max_log * FREQ_W +
        product_score["평균인기도"] / max_pop * POP_W
        if max_log > 0 and max_pop > 0 else 0
    )
    return product_score.sort_values("최종점수", ascending=False).head(top_n), used_level


def get_top3_combos(candidate_ids, space):
    candidate_posts = posts[posts["contentId"].isin(candidate_ids)].copy()
    candidate_posts = candidate_posts.sort_values("popularity_score", ascending=False)

    combos = []
    for _, post_row in candidate_posts.head(10).iterrows():
        cid = post_row["contentId"]
        post_tags = tags_valid[tags_valid["contentId"] == cid].copy()
        if space:
            post_tags = post_tags[post_tags["place_label"] == space]
        if len(post_tags) == 0:
            continue

        cats = post_tags["ohou_category_depth2"].value_counts().head(5).index.tolist()
        if len(cats) >= 2:
            combos.append({
                "contentId": cid,
                "categories": cats,
                "popularity": post_row["popularity_score"],
            })
        if len(combos) == 3:
            break

    return combos


# ══════════════════════════════════════════
# 이미지 유틸 함수
# ══════════════════════════════════════════
def image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def url_to_base64(url):
    try:
        resp = requests.get(url, timeout=10)
        img  = Image.open(BytesIO(resp.content)).convert("RGB")
        buf  = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return None


# ══════════════════════════════════════════
# GPT-4V 상품 설명 추출
# ══════════════════════════════════════════
def get_product_description(product_url, product_name):
    try:
        b64 = url_to_base64(product_url)
        if not b64:
            return product_name
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text",
                     "text": (
                         f"이 인테리어 상품 '{product_name}'을 자세히 설명해줘. "
                         f"색상, 재질, 형태, 크기감, 스타일을 구체적으로. "
                         f"3문장 이내로."
                     )}
                ]
            }],
            max_tokens=200
        )
        return response.choices[0].message.content
    except:
        return product_name


# ══════════════════════════════════════════
# AI 합성 함수
# ══════════════════════════════════════════
def synthesize_single(room_img, product_url, product_name, category):
    try:
        with st.spinner("상품 분석 중..."):
            product_desc = get_product_description(product_url, product_name)

        with st.spinner("방 사진 분석 중..."):
            room_b64 = base64.b64encode(image_to_bytes(room_img).getvalue()).decode()
            pos_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{room_b64}"}},
                        {"type": "text",
                         "text": (
                             f"이 방 사진에서 {category}가 있는 위치를 "
                             f"'왼쪽 하단', '중앙', '오른쪽' 등으로 간단히 설명해줘. "
                             f"없으면 '중앙'이라고 해줘. 한 단어로만."
                         )}
                    ]
                }],
                max_tokens=20
            )
            position = pos_response.choices[0].message.content.strip()

        with st.spinner("AI 합성 중..."):
            room_buf = image_to_bytes(room_img)
            prompt = (
                f"이 방 사진의 {position}에 있는 {category}를 "
                f"다음 상품으로 자연스럽게 교체해줘: {product_desc}. "
                f"방의 나머지 부분은 절대 바꾸지 말고 "
                f"조명과 원근감을 맞춰서 합성해줘."
            )
            response = client.images.edit(
                model="gpt-image-1",
                image=("room.png", room_buf, "image/png"),
                prompt=prompt,
                size="1024x1024",
            )
            result_b64 = response.data[0].b64_json
            return Image.open(BytesIO(base64.b64decode(result_b64)))

    except Exception as e:
        print(f"합성 에러: {e}")
        st.error(f"합성 오류: {e}")
        return None


def synthesize_full(room_img, products, space):
    try:
        product_descriptions = []
        for i, (cat, prod_row) in enumerate(products):
            with st.spinner(f"상품 분석 중... ({i+1}/{len(products)})"):
                desc = get_product_description(
                    str(prod_row["originalImageUrl"]),
                    str(prod_row["productName"])
                )
                zone_row = grid_representative[
                    (grid_representative["place_label"] == space) &
                    (grid_representative["top_category"] == cat)
                ]
                zone = zone_row["grid_zone"].values[0] \
                    if len(zone_row) > 0 else "중앙"
                product_descriptions.append(f"{zone}에 {cat}: {desc}")

        with st.spinner("AI가 방 전체를 꾸미는 중..."):
            room_buf = image_to_bytes(room_img)
            placement_str = " / ".join(product_descriptions)
            prompt = (
                f"이 방 사진을 인테리어 전문가처럼 꾸며줘. "
                f"다음 상품들을 지정된 위치에 자연스럽게 배치해줘: {placement_str}. "
                f"방의 구조(벽, 바닥, 천장)는 절대 바꾸지 말고 "
                f"각 상품이 실제로 그 위치에 있는 것처럼 조명과 원근감을 맞춰서 합성해줘."
            )
            response = client.images.edit(
                model="gpt-image-1",
                image=("room.png", room_buf, "image/png"),
                prompt=prompt,
                size="1024x1024",
            )
            result_b64 = response.data[0].b64_json
            return Image.open(BytesIO(base64.b64decode(result_b64)))

    except Exception as e:
        print(f"합성 에러: {e}")
        st.error(f"합성 오류: {e}")
        return None


# ══════════════════════════════════════════
# 상품 카드 렌더링 (방전체 — 선택 기능 포함)
# ══════════════════════════════════════════
def render_product_card_selectable(row, cat, btn_counter):
    """방전체용 — 상품 선택 버튼 포함"""
    is_selected = (
        cat in st.session_state["selected_products"] and
        st.session_state["selected_products"][cat]["productId"] == row["productId"]
    )
    with st.container(border=True):
        if pd.notna(row["originalImageUrl"]) and row["originalImageUrl"]:
            st.image(str(row["originalImageUrl"]), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x300?text=No+Image",
                     use_container_width=True)
        st.markdown(f"**{str(row['productName'])[:30]}**")
        st.caption(f"{row['brand']}")
        c1, c2 = st.columns(2)
        c1.metric("가격", f"{row['sellingPrice']:,.0f}원")
        c2.metric("점수", f"{row['최종점수']:.2f}")
        st.link_button("🔗 오늘의집", url=str(row["productUrl"]),
                       use_container_width=True)

        btn_key = f"select_{btn_counter[0]}"
        btn_counter[0] += 1
        btn_label = "✅ 선택됨" if is_selected else "이 상품 선택"
        if st.button(btn_label, key=btn_key,
                     use_container_width=True,
                     type="primary" if is_selected else "secondary"):
            st.session_state["selected_products"][cat] = row.to_dict()
            st.rerun()


def render_product_card(row, room_img, cat, btn_counter):
    """특정공간용 — 선택 + 방에 넣어보기 버튼 포함"""
    is_selected = (
        cat in st.session_state["selected_products"] and
        st.session_state["selected_products"][cat]["productId"] == row["productId"]
    )
    img_url = str(row["originalImageUrl"]) if pd.notna(row["originalImageUrl"]) and str(row["originalImageUrl"]).startswith("http") else None
    with st.container(border=True):
        if img_url:
            st.image(img_url, use_container_width=True)
        else:
            st.markdown(
                '<div style="height:160px;background:#f0f2f4;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#aab0b7;font-size:.85rem;">이미지 없음</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"**{str(row['productName'])[:30]}**")
        st.caption(f"{row['brand']}")
        c1, c2 = st.columns(2)
        c1.metric("가격", f"{row['sellingPrice']:,.0f}원")
        c2.metric("점수", f"{row['최종점수']:.2f}")
        st.link_button("🔗 오늘의집", url=str(row["productUrl"]),
                       use_container_width=True)

        sel_key = f"sel_{btn_counter[0]}"
        btn_counter[0] += 1
        sel_label = "✅ 선택됨" if is_selected else "이 상품 선택"
        if st.button(sel_label, key=sel_key, use_container_width=True,
                     type="primary" if is_selected else "secondary"):
            st.session_state["selected_products"][cat] = row.to_dict()
            st.rerun()

        if room_img is not None:
            synth_key = f"synth_{btn_counter[0]}"
            btn_counter[0] += 1
            if st.button("🪄 방에 넣어보기", key=synth_key,
                         use_container_width=True):
                base_img = st.session_state["synth_result"] \
                    if st.session_state["synth_result"] is not None \
                    else room_img
                result_img = synthesize_single(
                    base_img,
                    str(row["originalImageUrl"]),
                    str(row["productName"]),
                    str(row["ohou_category_depth2"])
                )
                if result_img is not None:
                    st.session_state["synth_result"]  = result_img
                    st.session_state["synth_product"] = str(row["productName"])
                    st.rerun()


# ══════════════════════════════════════════
# UI — 홈/필터
# ══════════════════════════════════════════
inject_dashboard_css()

st.markdown(
    """
    <div class="oh-header">
      <div class="oh-brand">
        <div class="oh-logo">
          <span class="oh-badge">
            <svg width="36" height="36" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <path d="M50 6 C44 6, 30 18, 16 34 C8 44, 4 54, 6 66 C8 78, 16 88, 28 92 C36 95, 64 95, 72 92 C84 88, 92 78, 94 66 C96 54, 92 44, 84 34 C70 18, 56 6, 50 6Z" fill="#35c5f0"/>
            </svg>
          </span>
          <span>오늘의집</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([2.4, 1], vertical_alignment="top")
with hero_left:
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero-slider">
            <div class="hero-track">
              <div class="hero-slide" style="background-image:url('https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1400&q=80');">
                <div class="hero-content">
                  <div class="hero-kicker">AI 맞춤 추천 대시보드</div>
                  <h1 class="hero-title">오늘의집 맞춤 상품 추천 대시보드</h1>
                  <p class="hero-copy">사용자 조건과 공간 성향을 기반으로 인테리어 상품과 공간 조합을 추천해드려요</p>
                </div>
              </div>
              <div class="hero-slide" style="background-image:url('https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=1400&q=80');">
                <div class="hero-content">
                  <div class="hero-kicker">AI 맞춤 추천 대시보드</div>
                  <h1 class="hero-title">방 전체 조합부터 단일 상품까지</h1>
                  <p class="hero-copy">조건 필터와 공간 유형 테스트를 결합해 더 정교한 추천 흐름을 제공합니다</p>
                </div>
              </div>
              <div class="hero-slide" style="background-image:url('https://images.unsplash.com/photo-1484154218962-a197022b5858?auto=format&fit=crop&w=1400&q=80');">
                <div class="hero-content">
                  <div class="hero-kicker">AI 맞춤 추천 대시보드</div>
                  <h1 class="hero-title">오늘의집 무드에 맞춘 탐색 경험</h1>
                  <p class="hero-copy">추천 결과, 오늘의집 링크, AI 합성까지 한 화면 흐름 안에서 이어집니다</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown('<div class="stat-panel"><h3>AI 분석 현황</h3>', unsafe_allow_html=True)
    stats = [
        ("추천 기준 수", "24개", ""),
        ("공간 유형 수", "16개", ""),
        ("스타일 카테고리", "12개", ""),
    ]
    for lbl, num, _ in stats:
        st.markdown(
            f"""
            <div class="hero-stat">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="hero-stat-lbl">{lbl}</div>
                <span class="hero-stat-num">{num}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    current_space_type = st.session_state.get("space_type", "")
    current_space_type_name = st.session_state.get("space_type_name", "")
    status_title = current_space_type_name if current_space_type else "공간인식유형 미확인"
    status_copy = "테스트 후 더 정확한 맞춤 추천을 받아요" if not current_space_type else "테스트 결과가 추천 필터에 반영됩니다"
    st.markdown(
        f"""
        <div style="margin-top:1rem; background:rgba(255,255,255,.92); border-radius:20px; padding:1rem 1.1rem; color:#2f3438;">
          <div style="font-weight:800; font-size:1rem;">{status_title}</div>
          <div style="margin-top:.25rem; color:#7b858e;">{status_copy}</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="guide-title">이런 사진 찾고 있나요?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="guide-copy">필터를 설정하면 맞춤 추천 결과와 공간 조합, 단일 상품 추천까지 한 번에 확인할 수 있어요.</div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("**1. 조건 설정**")
        st.caption("거주유형, 평수, 가족구성, 스타일, 톤, 예산, 공간 유형을 선택합니다.")
with c2:
    with st.container(border=True):
        st.markdown("**2. 추천 방식 선택**")
        st.caption("방전체, 특정공간, 단일상품 중 원하는 탐색 방식을 고릅니다.")
with c3:
    with st.container(border=True):
        st.markdown("**3. 결과 확인 및 합성**")
        st.caption("상품 추천, 조합 선택, 오늘의집 링크 이동, AI 방 합성 결과까지 이어집니다.")

st.markdown(
    """
    <div class="cta-strip">
      <div>
        <div class="cta-title">공간인식유형 테스트 하러가기</div>
        <p class="cta-copy">나의 공간 성향을 파악하면 더 정확한 맞춤 추천을 드려요</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
cta_cols = st.columns([5, 1.5])
with cta_cols[1]:
    st.markdown('<div style="margin-top:-3.2rem; margin-bottom:1.5rem;">', unsafe_allow_html=True)
    if st.button("🧪 테스트 시작 →", key="cta_test_btn", use_container_width=True):
        st.session_state["show_space_test"] = True
    st.markdown('</div>', unsafe_allow_html=True)

# ── 공간인식유형 테스트 (인라인) ──
if st.session_state.get("show_space_test"):
    import streamlit.components.v1 as components
    import json
    st.markdown("---")
    st.subheader("🏠 공간 인식 유형 테스트")
    st.caption("테스트를 완료한 후 아래에서 유형을 선택하세요.")

    TOP3_CSV = Path("/Users/SAMSUNG/Downloads/space_type_top3_urls.csv")
    POST_IMAGE_CSV = BASE / "housewarming_product_tags_with_image.csv"

    @st.cache_data(show_spinner=False)
    def _load_space_type_top3():
        top3 = pd.read_csv(TOP3_CSV)
        top3["contentId"] = top3["url"].str.extract(r"/projects/(\d+)").astype("Int64")
        needed_ids = set(top3["contentId"].dropna().astype(int).tolist())
        image_map = {}
        if needed_ids:
            cols = ["contentId", "postTitle", "originalImageUrl"]
            for chunk in pd.read_csv(POST_IMAGE_CSV, usecols=cols, chunksize=100_000, low_memory=False):
                matched = chunk[chunk["contentId"].isin(needed_ids)].dropna(subset=["originalImageUrl"]).drop_duplicates(subset=["contentId"])
                for row in matched.itertuples(index=False):
                    cid = int(row.contentId)
                    if cid not in image_map:
                        image_map[cid] = {"image_url": str(row.originalImageUrl), "post_title": str(row.postTitle) if pd.notna(row.postTitle) else ""}
                if needed_ids.issubset(image_map.keys()):
                    break
        result = {}
        for space_type, group in top3.groupby("space_type", sort=False):
            cards = []
            for row in group.sort_values("pop_score", ascending=False).itertuples(index=False):
                cid = int(row.contentId) if pd.notna(row.contentId) else None
                meta = image_map.get(cid, {})
                cards.append({"title": str(row.title), "url": str(row.url), "pop_score": float(row.pop_score), "image_url": meta.get("image_url", ""), "post_title": meta.get("post_title", "")})
            result[str(space_type)] = cards[:3]
        return result

    with open(BASE / "space_test_v2.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    top3_json = json.dumps(_load_space_type_top3(), ensure_ascii=False)
    html_content = html_content.replace("__TOP_POSTS_JSON__", top3_json, 1)
    components.html(html_content, height=900, scrolling=True)

    st.divider()
    st.subheader("테스트 결과를 선택해주세요")
    st.caption("위 테스트에서 나온 유형을 선택하면 상품 추천에 바로 반영돼요.")
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        default_space_type = st.session_state.get("space_type", "")
        default_label = SPACE_TYPE_OPTIONS.get(default_space_type, "선택 안 함")
        default_idx = list(SPACE_TYPE_OPTIONS.values()).index(default_label)
        selected_label = st.selectbox("나의 공간 인식 유형", list(SPACE_TYPE_OPTIONS.values()), index=default_idx, key="space_result_select")
        selected_type = [k for k, v in SPACE_TYPE_OPTIONS.items() if v == selected_label][0]
    with res_col2:
        st.write("")
        st.write("")
        if st.button("✨ 이 유형으로 추천받기", type="primary", use_container_width=True, disabled=not selected_type):
            st.session_state["space_type"] = selected_type
            st.session_state["space_type_name"] = selected_label
            st.session_state["show_space_test"] = False
            st.success(f"✅ {selected_label} 저장 완료! 아래에서 추천받기를 눌러보세요.")
            st.rerun()
    st.markdown("---")

st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
st.markdown('<div class="filter-title">조건 설정</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="filter-meta"><span>드롭다운 필터를 조합해 추천 결과를 좁혀보세요</span><span>조건을 모두 선택하면 추천받기 버튼으로 바로 실행할 수 있어요</span></div>',
    unsafe_allow_html=True,
)

top_filter_cols = st.columns(4)
with top_filter_cols[0]:
    residence = st.selectbox(
        "주거형태",
        ["아파트", "원룸&오피스텔", "빌라&연립", "단독주택", "기타"],
        index=None,
        placeholder="선택 해주세요",
        key="residence",
    )
with top_filter_cols[1]:
    area_range = st.selectbox(
        "평수",
        ["~10평", "10~20평", "20~30평", "30~40평", "40~50평", "50평+"],
        index=None,
        placeholder="선택 해주세요",
        key="area_range",
    )
with top_filter_cols[2]:
    family = st.selectbox(
        "가족형태",
        ["신혼부부", "싱글라이프", "아기가 있는 집", "취학 자녀가 있는 집",
         "부모님과 함께 사는 집", "기타"],
        index=None,
        placeholder="선택 해주세요",
        key="family",
    )
with top_filter_cols[3]:
    style = st.selectbox(
        "스타일",
        ["내추럴", "모던", "빈티지&레트로", "미니멀&심플", "유니크&믹스매치", "러블리&로맨틱"],
        index=None,
        placeholder="선택 해주세요",
        key="style",
    )

sub_filter_cols = st.columns(4)
with sub_filter_cols[0]:
    expertise = st.selectbox(
        "공사유형",
        ["리모델링", "홈스타일링", "부분공사", "건축"],
        index=None,
        placeholder="선택 해주세요",
        key="expertise",
    )
with sub_filter_cols[1]:
    default_space_type = st.session_state.get("space_type", "")
    default_label = SPACE_TYPE_OPTIONS.get(default_space_type, "선택 안 함")
    default_idx = list(SPACE_TYPE_OPTIONS.values()).index(default_label)
    space_type_label = st.selectbox(
        "공간유형",
        list(SPACE_TYPE_OPTIONS.values()),
        index=default_idx,
        key="space_type_label",
    )
    space_type = [k for k, v in SPACE_TYPE_OPTIONS.items() if v == space_type_label][0]
with sub_filter_cols[2]:
    tone = st.selectbox(
        "컬러/톤",
        ["무채색", "웜톤", "쿨톤", "그린톤"],
        index=None,
        placeholder="선택 해주세요",
        key="tone",
    )
with sub_filter_cols[3]:
    space = st.selectbox(
        "공간",
        VALID_PLACES,
        index=None,
        placeholder="선택 해주세요",
        key="space",
    )

third_filter_cols = st.columns(4)
with third_filter_cols[0]:
    budget = st.slider(
        "예산 (만원)",
        min_value=0,
        max_value=3000,
        value=500,
        step=50,
        key="budget",
    ) * 10000
with third_filter_cols[1]:
    agent = st.selectbox(
        "시공방식",
        ["전문가", "셀프•DIY", "반셀프"],
        index=None,
        placeholder="선택 해주세요",
        key="agent",
    )
with third_filter_cols[2]:
    change_type = st.selectbox(
        "변경범위",
        ["방전체", "특정공간", "단일상품"],
        key="change_type_select",
    )
with third_filter_cols[3]:
    uploaded_file = st.file_uploader(
        "방 사진 업로드",
        type=["jpg", "jpeg", "png"],
        key="room_upload",
    )
    if uploaded_file:
        room_img_pil = Image.open(uploaded_file).convert("RGB")
        buf = BytesIO()
        room_img_pil.save(buf, format="PNG")
        st.session_state["room_img_bytes"] = buf.getvalue()

action_cols = st.columns([1.2, 1.2, 3.6])
with action_cols[0]:
    run = st.button("추천받기", use_container_width=True, type="primary")
with action_cols[1]:
    if st.button("초기화", use_container_width=True):
        for key in [
            "residence", "area_range", "family", "style", "tone", "space",
            "budget", "agent", "expertise", "space_type_label", "change_type",
            "room_upload"
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
with action_cols[2]:
    st.caption("조건을 모두 선택하면 추천받기 버튼으로 바로 실행할 수 있어요")

st.markdown("</div>", unsafe_allow_html=True)

selected_cats = []
depth1_single = depth2_single = depth3_single = None

if change_type == "특정공간":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if space:
        default_cats = (
            tags_valid[tags_valid["place_label"] == space]["ohou_category_depth2"]
            .value_counts().head(5).index.tolist()
        )
        all_d2 = sorted(
            tags_cat[tags_cat["place_label"] == space]["ohou_category_depth2"]
            .dropna().unique().tolist()
        )
        selected_cats = st.multiselect(
            "카테고리 선택",
            options=all_d2,
            default=default_cats[:3],
            help="추천받고 싶은 세부 카테고리를 여러 개 고를 수 있습니다.",
        )
    else:
        st.info("특정공간 추천은 먼저 `공간` 필터를 선택해야 합니다.")
    st.markdown("</div>", unsafe_allow_html=True)
elif change_type == "단일상품":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    category_cols = st.columns(3)
    d1_list = sorted(tags_cat["ohou_category_depth1"].dropna().unique().tolist())
    with category_cols[0]:
        depth1_single = st.selectbox("카테고리 1단계", d1_list, index=None, placeholder="선택 해주세요")
    if depth1_single:
        d2_list = sorted(
            tags_cat[tags_cat["ohou_category_depth1"] == depth1_single]["ohou_category_depth2"]
            .dropna().unique().tolist()
        )
        with category_cols[1]:
            depth2_single = st.selectbox("카테고리 2단계", d2_list, index=None, placeholder="선택 해주세요")
    if depth2_single:
        d3_list = sorted(
            tags_cat[
                (tags_cat["ohou_category_depth1"] == depth1_single) &
                (tags_cat["ohou_category_depth2"] == depth2_single)
            ]["ohou_category_depth3"].dropna().unique().tolist()
        )
        with category_cols[2]:
            depth3_single = st.selectbox("카테고리 3단계", d3_list, index=None, placeholder="선택 해주세요")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    space_type = st.session_state.get("space_type", "") if "space_type_label" not in st.session_state else [k for k, v in SPACE_TYPE_OPTIONS.items() if v == st.session_state["space_type_label"]][0]

if "uploaded_file" not in locals():
    uploaded_file = None
if "run" not in locals():
    run = False

if st.session_state.get("room_img_bytes"):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**업로드한 방 사진**")
    room_preview = Image.open(BytesIO(st.session_state["room_img_bytes"])).convert("RGB")
    st.image(room_preview, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# 추천 실행
# ══════════════════════════════════════════
if run:
    filtered = posts.copy()
    if residence:  filtered = filtered[filtered["features_residence"] == residence]
    if area_range: filtered = filtered[filtered["area_range"] == area_range]
    if family:     filtered = filtered[filtered["family_main"] == family]
    if agent:      filtered = filtered[filtered["features_agent"] == agent]
    if expertise:  filtered = filtered[filtered["features_expertise"] == expertise]
    if space_type: filtered = filtered[filtered["space_type"] == space_type]
    if budget and space:
        budget_col = f"budget_{space}"
        if budget_col in filtered.columns:
            filtered = filtered[
                filtered[budget_col].isna() | (filtered[budget_col] <= budget)]

    if len(filtered) < 10:
        filtered = posts.copy()
        if residence:  filtered = filtered[filtered["features_residence"] == residence]
        if area_range: filtered = filtered[filtered["area_range"] == area_range]
        if family:     filtered = filtered[filtered["family_main"] == family]
        if agent:      filtered = filtered[filtered["features_agent"] == agent]
        if space_type: filtered = filtered[filtered["space_type"] == space_type]
    if len(filtered) < 10:
        filtered = posts.copy()
        if residence:  filtered = filtered[filtered["features_residence"] == residence]
        if area_range: filtered = filtered[filtered["area_range"] == area_range]
        if family:     filtered = filtered[filtered["family_main"] == family]
        if space_type: filtered = filtered[filtered["space_type"] == space_type]
    if len(filtered) < 10:
        filtered = posts.copy()
        if residence:  filtered = filtered[filtered["features_residence"] == residence]
        if area_range: filtered = filtered[filtered["area_range"] == area_range]
        if family:     filtered = filtered[filtered["family_main"] == family]
    if len(filtered) < 10:
        filtered = posts.copy()

    with st.spinner("유사 인테리어 사례 탐색 중..."):
        user_row = filtered.iloc[[0]].copy()
        if style:      user_row["style_main"] = style
        if tone:       user_row["tone_group"] = tone
        if space_type: user_row["space_type"] = space_type
        if budget and space: user_row[f"budget_{space}"] = budget
        if space:      user_row[f"has_{space}"] = 1

        candidate_ids = get_candidates(filtered, user_row)

    st.session_state["candidate_ids"]       = candidate_ids.tolist()
    st.session_state["filtered_len"]        = len(filtered)
    st.session_state["recommend_done"]      = True
    st.session_state["change_type_saved"]   = change_type
    st.session_state["space_saved"]         = space
    st.session_state["selected_cats_saved"] = selected_cats
    st.session_state["depth1_saved"]        = depth1_single
    st.session_state["depth2_saved"]        = depth2_single
    st.session_state["depth3_saved"]        = depth3_single
    st.session_state["synth_result"]        = None
    st.session_state["selected_combo_idx"]  = None
    st.session_state["selected_products"]   = {}  # 선택 초기화


# ══════════════════════════════════════════
# 결과 표시
# ══════════════════════════════════════════
if st.session_state["recommend_done"]:
    candidate_ids = st.session_state["candidate_ids"]
    change_type   = st.session_state["change_type_saved"]
    space         = st.session_state["space_saved"]
    selected_cats = st.session_state["selected_cats_saved"] or []
    depth1_single = st.session_state["depth1_saved"]
    depth2_single = st.session_state["depth2_saved"]
    depth3_single = st.session_state["depth3_saved"]

    room_img = None
    if st.session_state["room_img_bytes"]:
        room_img = Image.open(
            BytesIO(st.session_state["room_img_bytes"])).convert("RGB")

    btn_counter = [0]

    # ── 합성 결과 먼저 출력 ──
    if st.session_state["synth_result"] is not None:
        st.subheader("✨ AI 합성 결과")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**원본**")
            if room_img:
                st.image(room_img, use_container_width=True)
        with col2:
            st.markdown(f"**합성 결과** — {st.session_state['synth_product']}")
            st.image(st.session_state["synth_result"], use_container_width=True)

        buffer = BytesIO()
        st.session_state["synth_result"].save(buffer, format="PNG")
        st.download_button(
            "💾 결과 이미지 다운로드",
            data=buffer.getvalue(),
            file_name="interior_result.png",
            mime="image/png",
            use_container_width=True
        )
        if st.button("🔄 원본으로 초기화"):
            st.session_state["synth_result"] = None
            st.rerun()
        st.divider()

    st.caption(
        f"🔍 조건 매칭 게시글 {st.session_state['filtered_len']:,}건 → "
        f"유사 사례 {len(candidate_ids)}건 분석"
    )
    st.divider()

    # ── 방전체 ──
    if change_type == "방전체":
        st.subheader("🏠 방 전체 추천")
        target_space = space if space else "거실"

        combos = get_top3_combos(candidate_ids, target_space)

        if not combos:
            st.warning("조합을 찾을 수 없어요. 조건을 바꿔보세요.")
        else:
            # ── 조합 선택 ──
            st.markdown("### 📋 인테리어 조합 선택")
            st.caption("실제 인기 인테리어 사례 기반 TOP3 카테고리 조합이에요.")

            combo_cols = st.columns(3)
            for i, combo in enumerate(combos):
                with combo_cols[i]:
                    is_selected = st.session_state["selected_combo_idx"] == i
                    with st.container(border=True):
                        st.markdown(f"**조합 {i+1}**")
                        for cat in combo["categories"][:5]:
                            st.markdown(f"• {cat}")
                        btn_label = "✅ 선택됨" if is_selected else f"조합 {i+1} 선택"
                        if st.button(btn_label, key=f"combo_{i}",
                                     use_container_width=True,
                                     type="primary" if is_selected else "secondary"):
                            st.session_state["selected_combo_idx"] = i
                            st.session_state["selected_products"]  = {}
                            st.session_state["synth_result"]       = None
                            st.rerun()

            st.divider()

            selected_idx = st.session_state["selected_combo_idx"]
            if selected_idx is not None and selected_idx < len(combos):
                selected_combo = combos[selected_idx]
                st.markdown(f"### 🛋️ 조합 {selected_idx+1} — 상품 선택")
                st.caption("각 카테고리에서 원하는 상품 1개를 선택하세요. 선택 후 방 전체 합성하기를 눌러주세요.")

                # ── 카테고리별 상품 카드 (선택 기능) ──
                for cat in selected_combo["categories"]:
                    result, used_level = recommend_by_category(
                        candidate_ids, target_space, cat, top_n=4)
                    if not result.empty:
                        # 선택된 상품 표시
                        selected_prod = st.session_state["selected_products"].get(cat)
                        label = f"#### 📦 {cat}"
                        if used_level and used_level != cat:
                            label += f" *(확장: {used_level})*"
                        if selected_prod:
                            label += f"  ✅ **{str(selected_prod['productName'])[:20]}** 선택됨"
                        st.markdown(label)

                        cols = st.columns(4)
                        for i, (_, row) in enumerate(result.iterrows()):
                            with cols[i % 4]:
                                render_product_card_selectable(row, cat, btn_counter)
                        st.divider()

                # ── 선택 현황 + 합성 버튼 ──
                selected_products = st.session_state["selected_products"]
                total_cats = len(selected_combo["categories"])
                selected_count = len(selected_products)

                st.markdown(f"**선택 현황: {selected_count}/{total_cats}개 선택됨**")

                if selected_count > 0:
                    # 선택된 상품 요약
                    for cat, prod in selected_products.items():
                        st.caption(f"• {cat}: {str(prod['productName'])[:30]}")

                    if room_img:
                        if st.button("🪄 선택한 상품으로 방 전체 합성하기",
                                     type="primary", use_container_width=True):
                            # 선택된 상품으로 합성
                            products_for_synth = [
                                (cat, pd.Series(prod))
                                for cat, prod in selected_products.items()
                            ]
                            result_img = synthesize_full(
                                room_img, products_for_synth, target_space
                            )
                            if result_img is not None:
                                st.session_state["synth_result"]  = result_img
                                st.session_state["synth_product"] = f"조합 {selected_idx+1} 합성"
                                st.rerun()
                    else:
                        st.info("방 사진을 업로드하면 AI 합성이 가능해요.")
                else:
                    st.info("각 카테고리에서 상품을 선택해주세요.")

            else:
                st.info("위에서 원하는 조합을 선택해주세요.")

    # ── 특정공간 ──
    elif change_type == "특정공간":
        if not selected_cats:
            st.warning("카테고리를 1개 이상 선택해주세요.")
        else:
            st.subheader(f"🛋️ {space or '선택 공간'} 추천 상품")
            for cat in selected_cats:
                result, used_level = recommend_by_category(
                    candidate_ids, space or None, cat, top_n=4)
                if not result.empty:
                    label = f"### 📦 {cat}"
                    if used_level and used_level != cat:
                        label += f" *(확장: {used_level})*"
                    st.markdown(label)
                    cols = st.columns(4)
                    for i, (_, row) in enumerate(result.iterrows()):
                        with cols[i % 4]:
                            render_product_card(row, room_img, cat, btn_counter)
                    st.divider()
                else:
                    st.info(f"{cat}: 추천 결과가 없어요.")

    # ── 단일상품 ──
    elif change_type == "단일상품":
        if not depth1_single:
            st.warning("카테고리 1단계를 선택해주세요.")
        else:
            cat_label = " > ".join([x for x in [depth1_single, depth2_single,
                                                  depth3_single] if x])
            st.markdown(f"## 🔍 단일 상품 추천: {cat_label}")

            # 이미지 있는 상품만 대상으로
            candidate_tags = tags_valid[
                tags_valid["contentId"].isin(candidate_ids) &
                tags_valid["originalImageUrl"].notna() &
                (tags_valid["originalImageUrl"].astype(str).str.startswith("http"))
            ].copy()
            if space:
                candidate_tags = candidate_tags[
                    candidate_tags["place_label"] == space]

            # depth1 → depth2 → depth3 순으로 좁히기
            for depth_col, val in [
                ("ohou_category_depth3", depth3_single),
                ("ohou_category_depth2", depth2_single),
                ("ohou_category_depth1", depth1_single),
            ]:
                if val:
                    sub = candidate_tags[candidate_tags[depth_col] == val]
                    if len(sub) >= MIN_COUNT:
                        candidate_tags = sub
                        break

            # candidate_ids 범위 안에서 결과가 적으면 전체 tags_valid로 확장
            if len(candidate_tags) < MIN_COUNT:
                candidate_tags = tags_valid[
                    tags_valid["originalImageUrl"].notna() &
                    (tags_valid["originalImageUrl"].astype(str).str.startswith("http"))
                ].copy()
                for depth_col, val in [
                    ("ohou_category_depth3", depth3_single),
                    ("ohou_category_depth2", depth2_single),
                    ("ohou_category_depth1", depth1_single),
                ]:
                    if val:
                        sub = candidate_tags[candidate_tags[depth_col] == val]
                        if len(sub) >= 1:
                            candidate_tags = sub
                            break

            candidate_tags = candidate_tags.merge(
                posts[["contentId", "popularity_score"]], on="contentId", how="left"
            )
            product_score = (
                candidate_tags.groupby([
                    "productId", "productName", "brand", "sellingPrice",
                    "ohou_category_depth1", "ohou_category_depth2",
                    "originalImageUrl", "productUrl"
                ])
                .agg(등장수=("contentId","count"), 평균인기도=("popularity_score","mean"))
                .reset_index()
            )
            # 이미지 URL 확실히 있는 것만
            product_score = product_score[
                product_score["originalImageUrl"].notna() &
                (product_score["originalImageUrl"].astype(str).str.startswith("http"))
            ]
            # 등장수 2회 이상 우선, 없으면 1회 이상으로 완화
            ps_filtered = product_score[product_score["등장수"] >= 2]
            if len(ps_filtered) == 0:
                ps_filtered = product_score

            if len(ps_filtered) > 0:
                max_log = np.log1p(ps_filtered["등장수"]).max()
                max_pop = ps_filtered["평균인기도"].max()
                ps_filtered = ps_filtered.copy()
                ps_filtered["최종점수"] = (
                    np.log1p(ps_filtered["등장수"]) / max_log * FREQ_W +
                    ps_filtered["평균인기도"] / max_pop * POP_W
                    if max_log > 0 and max_pop > 0 else 0
                )
                result = ps_filtered.sort_values("최종점수", ascending=False).head(12)
                cat_key = depth2_single or depth1_single or ""
                cols = st.columns(4)
                for i, (_, row) in enumerate(result.iterrows()):
                    with cols[i % 4]:
                        render_product_card(row, room_img, cat_key, btn_counter)
            else:
                st.warning("추천 결과가 없어요. 카테고리를 바꿔보세요.")

else:
    pass
