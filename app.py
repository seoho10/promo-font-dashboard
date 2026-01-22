import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Font Recommender MVP", layout="wide")


FONTS_JSON_PATH = "assets/fonts/fonts.json"
DEFAULT_CANVAS_WIDTH = 1400
DEFAULT_CANVAS_HEIGHT = 900

# PC/MO 미리보기 사이즈 (나중에 수정 가능)
PC_PREVIEW_WIDTH = 1920
PC_PREVIEW_HEIGHT = 1080
MO_PREVIEW_WIDTH = 1080
MO_PREVIEW_HEIGHT = 1920


# -----------------------------
# Data models
# -----------------------------
@dataclass
class FontMeta:
    id: str
    name: str
    path: Optional[str]  # local font file path (ttf/otf). Can be None for MVP.
    tags: List[str]      # e.g. ["bold", "promo", "modern", "high_readability"]
    weights: List[int]   # e.g. [400, 700, 900]
    license: str         # e.g. "OFL", "Commercial", "Internal"


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_font_db(path: str) -> List[FontMeta]:
    """
    Load fonts.json. If missing, fallback to an in-code minimal sample.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raw = {
            "fonts": [
                {
                    "id": "pretendard",
                    "name": "Pretendard",
                    "path": None,
                    "tags": ["modern", "clean", "high_readability", "ecommerce"],
                    "weights": [400, 600, 700, 800],
                    "license": "OFL/OSS"
                },
                {
                    "id": "gmarket_sans",
                    "name": "Gmarket Sans",
                    "path": None,
                    "tags": ["bold", "promo", "energetic", "headline"],
                    "weights": [500, 700, 900],
                    "license": "OSS"
                },
                {
                    "id": "noto_serif_kr",
                    "name": "Noto Serif KR",
                    "path": None,
                    "tags": ["classic", "premium", "brand", "serif"],
                    "weights": [400, 600, 700],
                    "license": "OFL/OSS"
                }
            ]
        }

    fonts: List[FontMeta] = []
    for item in raw.get("fonts", []):
        fonts.append(
            FontMeta(
                id=item.get("id", ""),
                name=item.get("name", ""),
                path=item.get("path"),
                tags=item.get("tags", []),
                weights=item.get("weights", []),
                license=item.get("license", ""),
            )
        )
    return fonts


def _safe_open_image(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    # Convert to RGBA for safe compositing
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img.convert("RGBA")


def analyze_brief_dummy(headline: str, subcopy: str, intent: str) -> Dict:
    """
    Placeholder for AI analysis.
    Replace this later with LLM / CLIP.
    Output should be 'normalized brief' JSON used by scorer + preview.
    """
    text = f"{headline} {subcopy} {intent}".lower()

    # very simple heuristics for MVP
    tone = "promo" if any(k in text for k in ["sale", "세일", "%", "할인", "특가", "최대"]) else "brand"
    mood = "bold" if any(k in text for k in ["최대", "긴급", "오늘", "마감", "only", "now", "라스트"]) else "clean"
    target = "2030" if any(k in text for k in ["20", "30", "mz", "z세대"]) else "general"
    priority = ["readability", "attention"] if tone == "promo" else ["brand_fit", "readability"]

    return {
        "tone": tone,               # "promo" | "brand"
        "mood": mood,               # "bold" | "clean" | ...
        "target": target,           # "2030" | "general"
        "priority": priority,       # list[str]
        "avoid": ["thin", "decorative"] if tone == "promo" else [],
        "recommend_tags": [tone, mood, "high_readability"]
    }


def score_font(font: FontMeta, brief: Dict) -> float:
    """
    Deterministic scoring based on tag overlap + priorities.
    """
    want = set(brief.get("recommend_tags", []))
    have = set(font.tags)

    overlap = len(want & have)
    base = overlap * 10.0

    # small bonuses
    if "high_readability" in have:
        base += 6.0
    if brief.get("tone") == "promo" and ("headline" in have or "bold" in have):
        base += 6.0
    if brief.get("tone") == "brand" and ("premium" in have or "brand" in have):
        base += 6.0

    # penalties
    avoid = set(brief.get("avoid", []))
    if len(avoid & have) > 0:
        base -= 8.0

    return base


def rank_fonts(fonts: List[FontMeta], brief: Dict, topk: int = 3) -> List[Tuple[FontMeta, float]]:
    scored = [(f, score_font(f, brief)) for f in fonts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def pick_font_file_or_fallback(font: FontMeta, size_px: int) -> ImageFont.FreeTypeFont:
    """
    Attempt to load a real font file if path is provided.
    Otherwise fallback to PIL default font (so MVP still runs).
    """
    if font.path:
        try:
            return ImageFont.truetype(font.path, size=size_px)
        except Exception:
            pass
    # fallback
    return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> Tuple[ImageFont.ImageFont, int]:
    """
    Naive font sizing: if truetype, shrink until fits.
    For default font, just return.
    """
    # If default font, can't resize reliably
    if getattr(font, "path", None) is None and not hasattr(font, "font"):
        return font, 0

    size = getattr(font, "size", 48) if hasattr(font, "size") else 48
    while size > 12:
        try:
            test_font = ImageFont.truetype(getattr(font, "path", ""), size=size)  # may fail
        except Exception:
            return font, 0

        bbox = draw.textbbox((0, 0), text, font=test_font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            return test_font, size
        size -= 2
    return font, 0


def generate_preview(
    base_img: Image.Image,
    font_meta: FontMeta,
    headline: str,
    subcopy: str,
    layout: str,
    device_type: str = "pc",  # "pc" or "mo"
) -> Image.Image:
    """
    Generate a simple banner preview by placing text over the uploaded image.
    layout: "A" (headline top, offer center), "B" (offer big center), "C" (brand mood)
    device_type: "pc" or "mo" - determines preview dimensions
    """
    # Resize base image to target device size
    if device_type == "pc":
        target_w, target_h = PC_PREVIEW_WIDTH, PC_PREVIEW_HEIGHT
    else:  # "mo"
        target_w, target_h = MO_PREVIEW_WIDTH, MO_PREVIEW_HEIGHT
    
    # Resize base image maintaining aspect ratio, then crop/center to target size
    base_w, base_h = base_img.size
    base_ratio = base_w / base_h
    target_ratio = target_w / target_h
    
    if base_ratio > target_ratio:
        # Base is wider, fit to height
        new_h = target_h
        new_w = int(base_w * (target_h / base_h))
    else:
        # Base is taller, fit to width
        new_w = target_w
        new_h = int(base_h * (target_w / base_w))
    
    img_resized = base_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create target canvas and paste resized image centered
    img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    img.paste(img_resized, (paste_x, paste_y))
    
    W, H = target_w, target_h

    # overlay for contrast
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Darken bottom area slightly for readability
    odraw.rectangle([0, int(H * 0.55), W, H], fill=(0, 0, 0, 90))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    # Font sizes (rough defaults)
    headline_size = max(28, int(H * 0.08))
    subcopy_size = max(18, int(H * 0.045))

    headline_font = pick_font_file_or_fallback(font_meta, headline_size)
    subcopy_font = pick_font_file_or_fallback(font_meta, subcopy_size)

    margin_x = int(W * 0.06)
    max_text_w = int(W * 0.88)

    # positions
    if layout == "A":
        y1 = int(H * 0.62)
        y2 = int(H * 0.74)
    elif layout == "B":
        y1 = int(H * 0.64)
        y2 = int(H * 0.80)
        # make headline bigger in B
        headline_font = pick_font_file_or_fallback(font_meta, int(headline_size * 1.25))
    else:  # "C"
        y1 = int(H * 0.60)
        y2 = int(H * 0.76)

    # draw headline
    # (If you want auto-fit, implement robust sizing; keeping MVP simple & stable)
    draw.text((margin_x, y1), headline, font=headline_font, fill=(255, 255, 255, 255))
    if subcopy.strip():
        draw.text((margin_x, y2), subcopy, font=subcopy_font, fill=(255, 255, 255, 235))

    # small badge with font name
    badge_h = int(H * 0.055)
    badge_w = int(W * 0.32)
    bx, by = margin_x, int(H * 0.05)
    draw.rounded_rectangle([bx, by, bx + badge_w, by + badge_h], radius=14, fill=(0, 0, 0, 120))
    badge_font = ImageFont.load_default()
    draw.text((bx + 14, by + 12), f"Font: {font_meta.name} / Layout {layout}", font=badge_font, fill=(255, 255, 255, 220))

    return img


def img_to_bytes_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.title("기획전 메인 이미지 폰트 추천 MVP (Streamlit Skeleton)")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) 입력")
    uploaded = st.file_uploader("메인 이미지 업로드 (PNG/JPG)", type=["png", "jpg", "jpeg"])
    headline = st.text_input("메인 문구(헤드라인)", value="WINTER SALE 최대 70%")
    subcopy = st.text_input("서브 문구(선택)", value="아우터/패딩 특가, 기간 한정")
    intent = st.text_area("기획 의도(자유서술)", value="재고 소진이 목적이고, 긴급 구매 유도. 타겟은 20~30. 온라인 메인 배너.")

    st.divider()

    st.subheader("2) 폰트 DB")
    fonts = load_font_db(FONTS_JSON_PATH)
    st.caption(f"현재 로드된 폰트 수: {len(fonts)} (fonts.json 없으면 샘플 3개로 동작)")
    with st.expander("폰트 DB 미리보기"):
        st.json({"fonts": [f.__dict__ for f in fonts[:8]]})

    run_btn = st.button("분석 & 추천 생성", type="primary", use_container_width=True)


with right:
    st.subheader("결과")

    if not uploaded:
        st.info("좌측에서 이미지를 업로드하면, 추천 결과와 미리보기가 생성됩니다.")
    else:
        base_img = _safe_open_image(uploaded)
        st.image(base_img, caption="업로드된 원본 이미지", use_column_width=True)

    if run_btn:
        if not uploaded:
            st.error("이미지를 먼저 업로드해 주세요.")
            st.stop()

        # 1) AI 분석(현재는 더미)
        brief = analyze_brief_dummy(headline=headline, subcopy=subcopy, intent=intent)

        # 2) 폰트 추천
        top = rank_fonts(fonts, brief, topk=3)

        # 3) 결과 출력
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.subheader("AI 해석 브리프(구조화 결과)")
            st.json(brief)

            st.subheader("추천 폰트 TOP 3")
            for i, (fm, sc) in enumerate(top, start=1):
                st.markdown(f"**{i}. {fm.name}**  \n- score: `{sc:.1f}`  \n- tags: `{', '.join(fm.tags)}`  \n- license: `{fm.license}`")

        with c2:
            st.subheader("미리보기 시안 (폰트별 3가지 레이아웃)")
            for fm, sc in top:
                st.markdown(f"**{fm.name}**")
                
                # PC 버전
                st.markdown("##### PC 버전")
                prev_cols_pc = st.columns(3)
                for idx, layout in enumerate(["A", "B", "C"]):
                    preview_pc = generate_preview(
                        base_img=base_img,
                        font_meta=fm,
                        headline=headline,
                        subcopy=subcopy,
                        layout=layout,
                        device_type="pc",
                    )
                    png_bytes_pc = img_to_bytes_png(preview_pc)
                    with prev_cols_pc[idx]:
                        st.image(preview_pc, use_column_width=True)
                        st.download_button(
                            label=f"다운로드 ({fm.id}-{layout}-PC.png)",
                            data=png_bytes_pc,
                            file_name=f"{fm.id}-{layout}-PC.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl-pc-{fm.id}-{layout}",
                        )
                
                # MO 버전
                st.markdown("##### MO 버전")
                prev_cols_mo = st.columns(3)
                for idx, layout in enumerate(["A", "B", "C"]):
                    preview_mo = generate_preview(
                        base_img=base_img,
                        font_meta=fm,
                        headline=headline,
                        subcopy=subcopy,
                        layout=layout,
                        device_type="mo",
                    )
                    png_bytes_mo = img_to_bytes_png(preview_mo)
                    with prev_cols_mo[idx]:
                        st.image(preview_mo, use_column_width=True)
                        st.download_button(
                            label=f"다운로드 ({fm.id}-{layout}-MO.png)",
                            data=png_bytes_mo,
                            file_name=f"{fm.id}-{layout}-MO.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl-mo-{fm.id}-{layout}",
                        )
                
                st.divider()

        st.divider()
        st.caption("다음 단계: analyze_brief_dummy()를 LLM 호출로 교체 + (선택) 이미지 분석(CLIP) 추가 + fonts.json을 40~50개로 확장")


# -----------------------------
# Footer help
# -----------------------------
with st.expander("fonts.json 예시(복붙용)"):
    st.code(
        """{
  "fonts": [
    {
      "id": "pretendard",
      "name": "Pretendard",
      "path": "assets/fonts/Pretendard-Regular.ttf",
      "tags": ["modern", "clean", "high_readability", "ecommerce"],
      "weights": [400, 600, 700, 800],
      "license": "OFL/OSS"
    },
    {
      "id": "gmarket_sans",
      "name": "Gmarket Sans",
      "path": "assets/fonts/GmarketSansTTFBold.ttf",
      "tags": ["bold", "promo", "energetic", "headline", "high_readability"],
      "weights": [500, 700, 900],
      "license": "OSS"
    }
  ]
}""",
        language="json",
    )

st.caption("실행: `pip install -r requirements.txt` 후 `streamlit run app.py`")
