# pages/1_공간인식유형테스트.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from pathlib import Path

st.set_page_config(
    page_title="공간 인식 유형 테스트",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 새로고침 시 홈으로 이동 (의도적 진입만 허용) ──
if st.session_state.pop("allow_test_page", False):
    st.session_state["_on_test_page"] = True
elif not st.session_state.get("_on_test_page"):
    st.switch_page("pages/0_dashboard_main.py")

# ── 유형 저장 후 돌아온 경우 (쿠키 감지) ──
try:
    _cookie = st.context.cookies.get("space_type_result", "")
except Exception:
    _cookie = ""
if _cookie:
    st.session_state.pop("_on_test_page", None)
    st.switch_page("pages/0_dashboard_main.py")

# Streamlit UI 숨기기
st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {display:none !important;}
    header[data-testid="stHeader"] {display:none !important;}
    .block-container {padding:0 !important; max-width:100% !important;}
    iframe {border:none !important;}
    </style>
    <script>window.scrollTo(0,0);</script>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).resolve().parent.parent
TOP3_CSV = Path("/Users/SAMSUNG/Downloads/space_type_top3_urls.csv")
POST_IMAGE_CSV = BASE / "housewarming_product_tags_with_image.csv"

if "space_type" not in st.session_state:
    st.session_state["space_type"] = ""
if "space_type_name" not in st.session_state:
    st.session_state["space_type_name"] = ""


@st.cache_data(show_spinner=False)
def load_space_type_top3():
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
    html_raw = f.read()

top3_json = json.dumps(load_space_type_top3(), ensure_ascii=False)
html_raw = html_raw.replace("__TOP_POSTS_JSON__", top3_json, 1)

# ── 오늘의집 배지 제거 ──
html_raw = html_raw.replace(
    '<div class="brand-chip"><span class="brand-dot"></span>오늘의집 무드 UI</div>',
    '',
)

# ── PC 뷰 max-width 확대 ──
html_raw = html_raw.replace(
    '.inner{max-width:560px;',
    '.inner{max-width:720px;',
)
html_raw = html_raw.replace(
    'max-width:520px;',
    'max-width:680px;',
)

# ── 배경/레이아웃 수정 ──
html_raw = html_raw.replace(
    'min-height:100vh;',
    'min-height:auto;',
)

# ── CSS / 버튼 / JS 주입 ──
inject_css = """
/* 배경 끊김 방지 */
html, body {
  height:auto !important;
  min-height:100% !important;
  overflow:visible !important;
  background: #f5fbfd !important;
  background-image: none !important;
}
.page { min-height:auto !important; background:transparent !important; }
#p-intro, #p-survey, #p-loading, #p-result { background:transparent !important; }
#p-intro::before, #p-intro::after { display:none !important; }

/* 저장 완료 오버레이 */
.toast-overlay{
  position:fixed; inset:0; z-index:99999;
  background:rgba(0,0,0,.45);
  display:flex; align-items:center; justify-content:center;
  animation:fadeIn .25s ease;
}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
.toast-card{
  background:#fff; border-radius:20px;
  padding:2.5rem 2rem 2rem; text-align:center;
  box-shadow:0 20px 60px rgba(0,0,0,.25);
  font-family:'Noto Sans KR',sans-serif;
  animation:popUp .3s ease;
  max-width:340px; width:90%;
}
@keyframes popUp{from{opacity:0;transform:scale(.9);}to{opacity:1;transform:scale(1);}}
.toast-check{font-size:2.5rem; margin-bottom:.6rem;}
.toast-title{font-size:1.1rem; color:#24323d; margin-bottom:.3rem;}
.toast-sub{font-size:.9rem; color:#8a9aa5; margin-bottom:1.4rem;}
.toast-home-btn{
  display:block; width:100%; padding:1rem;
  border-radius:14px; border:none;
  background:linear-gradient(135deg,#35c5f0,#09addb);
  color:#fff !important; font-size:1.05rem; font-weight:800;
  cursor:pointer; font-family:'Noto Sans KR',sans-serif;
  text-decoration:none !important;
  box-shadow:0 10px 24px rgba(53,197,240,.24);
  transition:transform .15s;
  text-align:center;
  box-sizing:border-box;
}
.toast-home-btn:hover{transform:translateY(-1px);}

/* 뒤로가기 */
.back-btn{
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.7rem 1.2rem; border-radius:12px;
  background:#fff; border:1px solid #dce9ee;
  color:#24323d; font-size:.9rem; font-weight:700;
  cursor:pointer; font-family:'Noto Sans KR',sans-serif;
  box-shadow:0 4px 14px rgba(0,0,0,.08);
  text-decoration:none;
  transition:background .15s;
  margin:1.2rem 0 0 2rem;
}
.back-btn:hover{background:#eaf7fb;}

/* 추천받기 */
.apply-btn{
  display:block; width:100%; padding:1.1rem;
  border-radius:16px; border:none;
  background:linear-gradient(135deg,#35c5f0,#09addb);
  color:#fff; font-size:1.05rem; font-weight:800;
  cursor:pointer; font-family:'Noto Sans KR',sans-serif;
  margin-top:1rem;
  box-shadow:0 14px 26px rgba(53,197,240,.24);
  transition:transform .15s, box-shadow .15s;
}
.apply-btn:hover{transform:translateY(-1px);box-shadow:0 18px 30px rgba(53,197,240,.28);}
"""

inject_back_btn = ''

inject_apply_btn = """
      <button class="apply-btn" onclick="applyType()">✨ 이 유형으로 추천받기</button>
"""

inject_apply_js = """
function goHome(){
  try {
    var s = window.parent.document.createElement('script');
    s.textContent = 'window.location.href = "/";';
    window.parent.document.body.appendChild(s);
  } catch(e){
    try { window.top.location.href = '/'; } catch(e2){}
  }
}
function applyType(){
  var scores = calcScores();
  var code = classifyType(scores);
  var info = TYPES[code] || {};
  var name = info.name || '';
  document.cookie = 'space_type_result=' + code + '|' + name + ';path=/;max-age=300';
  var overlay = document.createElement('div');
  overlay.className = 'toast-overlay';
  overlay.innerHTML =
    '<div class="toast-card">' +
      '<div class="toast-check">✅</div>' +
      '<div class="toast-title"><b>' + code + ' — ' + name + '</b></div>' +
      '<div class="toast-sub">유형이 저장되었습니다</div>' +
      '<button class="toast-home-btn" onclick="goHome()">🏠 홈으로 이동하기</button>' +
    '</div>';
  document.body.appendChild(overlay);
}
"""

# 페이지 전환 시 맨 위로 스크롤
inject_scroll_top = """
function scrollParentTop(){
  try { window.parent.scrollTo(0,0); } catch(e){}
  try { window.top.scrollTo(0,0); } catch(e){}
}
var _origShow = show;
show = function(id){
  _origShow(id);
  scrollParentTop();
};
scrollParentTop();
"""

# CSS 주입
html_raw = html_raw.replace('</style>', inject_css + '\n</style>')

# 뒤로가기 버튼 (body 시작 직후)
html_raw = html_raw.replace('<body>', '<body>\n' + inject_back_btn)

# 추천받기 버튼 (결과 페이지 버튼 영역 뒤)
html_raw = html_raw.replace(
    """<button class="r-btn dark" onclick="downloadStoryImage()">스토리 저장</button>
    </div>""",
    """<button class="r-btn dark" onclick="downloadStoryImage()">스토리 저장</button>
    </div>""" + inject_apply_btn,
)

# JS 주입
html_raw = html_raw.replace('</script>', inject_scroll_top + '\n' + inject_apply_js + '\n</script>')

# ── Streamlit 홈 버튼 (iframe 밖 — 항상 동작) ──
_home_col, _ = st.columns([1, 5])
with _home_col:
    if st.button("← 홈으로", key="st_home_btn"):
        st.session_state.pop("_on_test_page", None)
        st.switch_page("pages/0_dashboard_main.py")

components.html(html_raw, height=3000, scrolling=False)
