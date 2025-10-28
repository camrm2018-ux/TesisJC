# streamlit_outs_variant_integrado.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import io as _io
from pathlib import Path
import time

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from skimage.exposure import equalize_adapthist
from skimage.util import img_as_float

# --------- Streamlit and canvas ---------
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# --------- PIL para lectura de imagen ---------
from PIL import Image as PILImage, Image

# ========================= CONSTANTES =========================
OUT1_STD_AREA_PX = 1_025_102  # OUT1 estandarizado (área de referencia del histograma)

APP_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Candidatos de nombres para carga automática
DEFAULT_DB_CSV_CANDIDATES = [
    "Base De Datos.csv", "BaseDeDatos.csv", "BD_Velocidad_Rugosidad.csv"
]
DEFAULT_OUTS_CSV_CANDIDATES = [
    "Tabla Outs.csv", "TablaOuts.csv", "outs.csv"
]

# ROI fija arrastrable
ROI_H_PX = 2314   # alto ROI
ROI_W_PX = 443    # ancho ROI

# Límites de visualización del canvas
MAX_CANVAS_W = 700
MAX_CANVAS_H = 1200

# ========================= UTILIDADES DE ARCHIVO =========================
def _normalize_name(s: str) -> str:
    s = (s or "").replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s.casefold()

def _find_first_existing(candidates: list[str]) -> Path | None:
    """
    Busca el primer archivo existente en DATA_DIR y APP_DIR (directo y recursivo),
    con coincidencia tolerante en nombre (ignora mayúsculas y espacios).
    """
    search_dirs = [DATA_DIR, APP_DIR]
    norm_targets = [_normalize_name(x) for x in candidates]

    # 1) Búsqueda directa
    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.exists() and p.is_file():
                return p

    # 2) No recursiva tolerante
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.csv"):
            if _normalize_name(p.name) in norm_targets:
                return p

    # 3) Recursiva tolerante
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.csv"):
            if _normalize_name(p.name) in norm_targets:
                return p
    return None

@st.cache_data(show_spinner=False)
def _read_csv_robusto_from_path(path: Path) -> pd.DataFrame | None:
    """Lee CSV desde ruta con separador/decimal desconocido."""
    if path is None or not path.exists():
        return None
    intentos = [
        dict(sep=";", decimal=","),
        dict(sep=";", decimal="."),
        dict(sep=",", decimal=","),
        dict(sep=",", decimal="."),
        dict(sep=None, engine="python"),
    ]
    for kw in intentos:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", **kw)
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
            return df
        except Exception:
            continue
    return None

# ========================= IMPORTACIÓN ROBUSTA DE BD EXTERNA =========================
@st.cache_data(show_spinner=False)
def read_csv_robusto(uploaded_file) -> pd.DataFrame | None:
    """Intenta leer CSV con separadores/decimales comunes (cacheado)."""
    if uploaded_file is None:
        return None
    intentos = [
        dict(sep=";", decimal=","),
        dict(sep=";", decimal="."),
        dict(sep=",", decimal=","),
        dict(sep=",", decimal="."),
        dict(sep=None, engine="python"),
    ]
    for kw in intentos:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig", **kw)
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
            return df
        except Exception:
            continue
    return None

def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

@st.cache_data(show_spinner=False)
def normalize_external_db(df_ext_raw: pd.DataFrame) -> pd.DataFrame | None:
    """Normaliza CSV a: Material, Velocidad, Rugosidad (mm)."""
    if df_ext_raw is None or df_ext_raw.empty:
        return None
    ext = df_ext_raw.copy()
    cols = {c.lower(): c for c in ext.columns}

    if "material" in cols and cols["material"] != "Material":
        ext = ext.rename(columns={cols["material"]: "Material"})
    if "velocidad" in cols and cols["velocidad"] != "Velocidad":
        ext = ext.rename(columns={cols["velocidad"]: "Velocidad"})
    elif "va" in cols:
        ext = ext.rename(columns={cols["va"]: "Velocidad"})
    if "rugosidad" not in [c.lower() for c in ext.columns]:
        if "rugosidad_mm" in cols:
            ext = ext.rename(columns={cols["rugosidad_mm"]: "Rugosidad"})
        elif "rugosidad_um" in cols:
            ext["Rugosidad"] = _to_num(ext[cols["rugosidad_um"]]) / 1000.0
    else:
        for c in list(ext.columns):
            if c.lower() == "rugosidad" and c != "Rugosidad":
                ext = ext.rename(columns={c: "Rugosidad"})

    if not set(["Material", "Velocidad", "Rugosidad"]).issubset(ext.columns):
        return None

    ext["Velocidad"] = _to_num(ext["Velocidad"])
    ext["Rugosidad"] = _to_num(ext["Rugosidad"])
    ext["Material"] = ext["Material"].astype(str).str.strip()
    ext = ext.dropna(subset=["Material", "Velocidad", "Rugosidad"])
    return ext[["Material", "Velocidad", "Rugosidad"]]

@st.cache_data(show_spinner=False)
def build_reference_db(df_ext_norm: pd.DataFrame | None = None):
    """BD interna de respaldo; si hay df_ext_norm se concatena para comparación."""
    material = "Aluminio"
    data = []
    for v in [2.928, 2.879, 2.945, 2.914, 2.938]:
        data.append({"Material": material, "Velocidad": np.nan, "Rugosidad": v})
    for v in [4.697, 4.600, 4.951, 4.831, 5.037, 4.798, 5.244]:
        data.append({"Material": material, "Velocidad": 280.0, "Rugosidad": v})
    for v in [5.072, 4.795, 4.973, 4.625, 5.000]:
        data.append({"Material": material, "Velocidad": 360.0, "Rugosidad": v})
    for v in [5.120, 5.132, 5.871, 4.942, 4.757]:
        data.append({"Material": material, "Velocidad": 890.0, "Rugosidad": v})
    df = pd.DataFrame(data)
    if df_ext_norm is not None and not df_ext_norm.empty:
        df = pd.concat([df, df_ext_norm], ignore_index=True)
    return df

# ========================= UI / Canvas helpers =========================
def _to_pil_rgb(image):
    """Convierte a PIL RGB (evita copias innecesarias)."""
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(arr)
    return PILImage.open(_io.BytesIO(image)).convert("RGB")

def compute_display_scale(w0, h0, max_w=MAX_CANVAS_W, max_h=MAX_CANVAS_H) -> float:
    """Escala de visualización para que quepa en el canvas sin ampliar."""
    return min(1.0, (max_w / float(w0)) if max_w else 1.0, (max_h / float(h0)) if max_h else 1.0)

def _make_fixed_rect(left_can: int, top_can: int, w_can: int, h_can: int) -> dict:
    """Crea el objeto rect fijo para st_canvas (transform), sin controles de escala/rotación."""
    return {
        "type": "rect",
        "left": int(left_can), "top": int(top_can),
        "width": int(w_can), "height": int(h_can),
        "fill": "rgba(0,255,0,0.15)", "stroke": "#00FF00", "strokeWidth": 2,
        "selectable": True, "evented": True,
        "hasControls": False, "lockScalingX": True, "lockScalingY": True, "lockRotation": True,
        "angle": 0, "scaleX": 1, "scaleY": 1,
    }

def canvas_with_scale(img_bgr, key, mode="rect", stroke_width=3,
                      max_canvas_w=MAX_CANVAS_W, max_canvas_h=MAX_CANVAS_H):
    """Ajusta la imagen al canvas sin cortes ni ampliación."""
    pil_bg = _to_pil_rgb(img_bgr)
    w0, h0 = pil_bg.size
    s_img = compute_display_scale(w0, h0, max_canvas_w, max_canvas_h)
    if s_img < 1.0:
        pil_bg = pil_bg.resize((int(round(w0*s_img)), int(round(h0*s_img))), resample=PILImage.BILINEAR)

    canvas_result = st_canvas(
        fill_color=("rgba(0,255,0,0.15)" if mode == "rect" else "rgba(0,0,0,0)"),
        stroke_width=stroke_width,
        stroke_color=("#00FF00" if mode == "rect" else "#FF0000"),
        background_image=pil_bg,
        height=pil_bg.size[1],
        width=pil_bg.size[0],
        drawing_mode=("rect" if mode == "rect" else "point"),
        key=key,
        update_streamlit=True,
    )
    return canvas_result, s_img

def detect_overexposed(gray: np.ndarray) -> float:
    """Porcentaje de píxeles saturados (>=245)."""
    return float(np.mean(gray >= 245) * 100.0)

# ========================= Filtros (Cap. 7) y OUTs =========================
MASK1 = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)

MASK2 = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
], dtype=np.float32)

def aplicar_filtros_cap7(Iroi):
    I = Iroi.astype(np.float32, copy=False)
    k1_sum = float(MASK1.sum()) if MASK1.sum() > 0 else 1.0
    f1 = cv2.filter2D(I, cv2.CV_32F, MASK1) / k1_sum
    return cv2.filter2D(f1, cv2.CV_32F, MASK2)

def to_uint8_display(I):
    J = I - np.min(I)
    rng = np.max(J) + 1e-9
    return np.clip((J / rng) * 255.0, 0, 255).astype(np.uint8)

def analisis_histograma(Iproc_uint8, ref_area_px=OUT1_STD_AREA_PX, smooth=1):
    """
    Histograma normalizado al área de referencia — produce OUT1, OUT2, OUT3.
    """
    hist = cv2.calcHist([Iproc_uint8], [0], None, [256], [0, 256]).ravel()
    if smooth and smooth > 0:
        k = max(1, int(smooth))
        hist = np.convolve(hist, np.ones(k, dtype=float) / k, mode="same")

    area_roi = float(Iproc_uint8.shape[0] * Iproc_uint8.shape[1])
    scale = float(ref_area_px) / max(area_roi, 1.0)
    hist_scaled = hist * scale

    out1 = int(round(hist_scaled.sum()))
    out2 = int(round(hist_scaled.max()))
    out3 = float(out1) / max(out2, 1e-9)
    return hist_scaled, out1, out2, out3

# ========================= Rangos/centroides desde OUTS =========================
@st.cache_data(show_spinner=False)
def compute_rangos_y_centroides_from_outs(df_outs: pd.DataFrame, q_low=0.05, q_high=0.95):
    """
    Espera columnas EXACTAS: ['Out1','Out2','Out3','Velocidad','Ra_bd'].
    Devuelve rangos 5–95% y centroides por clase de Ra_bd.
    """
    expected = ["Out1", "Out2", "Out3", "Velocidad", "Ra_bd"]
    if df_outs is None or df_outs.empty or any(c not in df_outs.columns for c in expected):
        return None, None

    d = df_outs.dropna(subset=["Ra_bd", "Out1", "Out2", "Out3"]).copy()
    if d.empty:
        return None, None

    def _q(group, col, q):
        return float(np.quantile(group[col].values, q))

    recs, meds = [], []
    for ra, g in d.groupby("Ra_bd"):
        g = g.copy()
        recs.append({
            "Ra_bd": float(ra),
            "n_muestras": int(len(g)),
            "Out1_q05": _q(g, "Out1", q_low),
            "Out1_q95": _q(g, "Out1", q_high),
            "Out2_q05": _q(g, "Out2", q_low),
            "Out2_q95": _q(g, "Out2", q_high),
            "Out3_q05": _q(g, "Out3", q_low),
            "Out3_q95": _q(g, "Out3", q_high),
        })
        meds.append({
            "Ra_bd": float(ra),
            "Out1_med": float(np.median(g["Out1"].values)),
            "Out2_med": float(np.median(g["Out2"].values)),
            "Out3_med": float(np.median(g["Out3"].values)),
        })

    rangos_df = pd.DataFrame.from_records(recs).sort_values("Ra_bd").reset_index(drop=True)
    centroides_df = pd.DataFrame.from_records(meds).sort_values("Ra_bd").reset_index(drop=True)
    return rangos_df, centroides_df

# ========================= Predicción robusta continua por OUTs =========================
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-12

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cw = np.cumsum(w) / (np.sum(w) + 1e-12)
    i = np.searchsorted(cw, 0.5)
    i = min(max(i, 0), len(v)-1)
    return float(v[i])

def robust_knn_ra(df_outs: pd.DataFrame, o1: float, o2: float, o3: float,
                  k: int | None = None, use_mahal: bool = True) -> tuple[float, dict]:
    """
    Predicción continua Ra usando kNN robusto:
      - Escalado robusto (mediana/MAD) de OUT1/2/3
      - Distancia de Mahalanobis en espacio escalado (fallback a Euclidiana si falla)
      - Ponderación w = 1/(d^2 + eps)
      - Estimador: mediana ponderada + blend con media ponderada
    Retorna (pred, info)
    """
    expected = ["Out1", "Out2", "Out3", "Ra_bd"]
    d = df_outs.dropna(subset=expected).copy()
    if d.empty:
        return np.nan, {"metodo": "knn_robusto", "detalle": "sin_datos"}

    X = d[["Out1", "Out2", "Out3"]].to_numpy(dtype=float)
    y = d["Ra_bd"].to_numpy(dtype=float)

    # Escalado robusto por columna
    meds = np.median(X, axis=0)
    mads = np.array([_mad(X[:, i]) for i in range(3)])
    Z = (X - meds) / mads
    z0 = (np.array([o1, o2, o3], dtype=float) - meds) / mads

    # Distancias
    eps = 1e-9
    try:
        S = np.cov(Z, rowvar=False)
        S_inv = np.linalg.pinv(S)
        diff = Z - z0
        dists = np.sqrt(np.einsum("ij,jk,ik->i", diff, S_inv, diff))  # Mahalanobis
        dist_kind = "mahalanobis"
    except Exception:
        dists = np.sqrt(np.sum((Z - z0)**2, axis=1))  # Euclidiana en Z
        dist_kind = "euclidiana"

    # k automático si no se pasa: ~sqrt(n), mínimo 5, máximo 25% del dataset
    n = len(dists)
    if k is None:
        k = int(np.clip(np.sqrt(n), 5, max(5, int(0.25*n))))
    idx = np.argsort(dists)[:k]
    dk = dists[idx]
    yk = y[idx]

    # Pesos: 1/(d^2 + eps), con amortiguación por percentil 75 para evitar dominancia de un punto
    d75 = np.percentile(dk, 75) + eps
    w = 1.0 / ((dk / d75)**2 + eps)

    # Estimadores
    y_med_w = _weighted_median(yk, w)
    y_mean_w = float(np.sum(w * yk) / (np.sum(w) + 1e-12))
    # Blend robusto (más peso a la mediana si distribución dispersa)
    spread = float(np.sqrt(np.sum(w * (yk - y_mean_w)**2) / (np.sum(w) + 1e-12)))
    alpha = 0.7 if spread > 0 else 0.5
    y_pred = alpha * y_med_w + (1 - alpha) * y_mean_w

    info = {
        "metodo": "knn_robusto",
        "distancia": dist_kind,
        "k": int(k),
        "vecinos": pd.DataFrame({
            "Ra_vecino": yk,
            "dist": dk,
            "peso": w
        }).sort_values("dist").reset_index(drop=True),
        "y_mediana_ponderada": float(y_med_w),
        "y_media_ponderada": float(y_mean_w),
        "spread_w": float(spread)
    }
    return float(y_pred), info

def clasificar_o_predecir_ra(out1: float, out2: float, out3: float,
                             df_outs: pd.DataFrame,
                             rangos_df: pd.DataFrame | None,
                             centroides_df: pd.DataFrame | None) -> tuple[float, dict]:
    """
    Híbrido:
      1) Si cae dentro de exactamente UNA clase por rangos 5–95% => usa ese Ra_bd (discreto).
      2) En otro caso => usa kNN robusto continuo (siempre devuelve valor).
    """
    x = np.array([out1, out2, out3], dtype=float)

    # Opción 1: intervalos 5–95% (si disponibles)
    if rangos_df is not None and not rangos_df.empty:
        cands = []
        for _, r in rangos_df.iterrows():
            ok1 = r["Out1_q05"] <= x[0] <= r["Out1_q95"]
            ok2 = r["Out2_q05"] <= x[1] <= r["Out2_q95"]
            ok3 = r["Out3_q05"] <= x[2] <= r["Out3_q95"]
            if ok1 and ok2 and ok3:
                cands.append(float(r["Ra_bd"]))
        if len(cands) == 1:
            return cands[0], {"metodo": "intervalos_5_95", "candidatos": cands}

    # Opción 2: kNN robusto continuo
    ra_pred, info_knn = robust_knn_ra(df_outs, out1, out2, out3, k=None, use_mahal=True)
    return ra_pred, info_knn

# ========================= Interfaz =========================
st.title("Estimación acabado superficial con visión artificial")

# ---- Carga automática de CSVs ----
auto_db_path = _find_first_existing(DEFAULT_DB_CSV_CANDIDATES)
auto_outs_path = _find_first_existing(DEFAULT_OUTS_CSV_CANDIDATES)

# ---- Cargar BD externa (referencia) ----
st.sidebar.header("Base de datos externa (Velocidad/Rugosidad)")
st.sidebar.caption("Puedes subir un CSV; si no, se carga automáticamente si existe en DATA_DIR/APP_DIR.")

csv_up_db = st.sidebar.file_uploader("Sube CSV (Material, Velocidad, Rugosidad)", type=["csv"], key="db_up")
if csv_up_db is not None:
    df_ext_raw = read_csv_robusto(csv_up_db)
    db_source_label = "Archivo subido"
else:
    df_ext_raw = _read_csv_robusto_from_path(auto_db_path) if auto_db_path else None
    db_source_label = f"Carga automática: {auto_db_path.name}" if auto_db_path else "Sin BD detectada"

df_norm_for_ui = normalize_external_db(df_ext_raw) if df_ext_raw is not None else None

if df_ext_raw is None or df_norm_for_ui is None or df_norm_for_ui.empty:
    st.sidebar.warning("BD no disponible o con columnas inválidas. (Se puede continuar con la BD interna).")
else:
    st.sidebar.success(f"BD cargada ({db_source_label})")
    with st.sidebar.expander("Vista rápida BD", expanded=False):
        st.dataframe(df_norm_for_ui.head(20), use_container_width=True)

# Construye BD con fallback interno
REF_DB = build_reference_db(df_norm_for_ui)

with st.sidebar:
    st.header("Selección de Material y Velocidad")
    materiales = sorted(REF_DB["Material"].dropna().unique().tolist())
    material_sel = st.selectbox("Material", materiales, index=0 if materiales else None)

    if (df_norm_for_ui is not None) and (not df_norm_for_ui.empty):
        sub_ext = df_norm_for_ui[df_norm_for_ui["Material"] == material_sel]
        vel_opts = sorted(sub_ext["Velocidad"].dropna().unique().tolist())
    else:
        sub_bd = REF_DB[REF_DB["Material"] == material_sel]
        vel_opts = sorted(sub_bd["Velocidad"].dropna().unique().tolist())

    vel_sel = st.selectbox("Velocidad (desde BD)", vel_opts, index=0 if vel_opts else None, key="vel_sel")

# ---- Cargar CSV de OUTS (solo lectura) ----
st.sidebar.header("CSV de OUTS")
st.sidebar.caption("Puedes subir un CSV; si no, se carga automáticamente si existe en DATA_DIR/APP_DIR.")

outs_csv_up = st.sidebar.file_uploader("Importa CSV OUTS", type=["csv"], key="outs_csv_up")

def _read_outs_csv_filelike(filelike) -> pd.DataFrame | None:
    if filelike is None:
        return None
    from io import StringIO
    try:
        filelike.seek(0)
        raw = filelike.read()
        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        df = pd.read_csv(StringIO(text), sep=None, engine="python")
    except Exception:
        return None
    expected = ["Out1", "Out2", "Out3", "Velocidad", "Ra_bd"]
    df.columns = [c.strip() for c in df.columns]
    if set(df.columns) == set(expected):
        df = df[expected]
    elif df.columns.tolist() != expected:
        st.sidebar.error(f"El CSV debe tener columnas EXACTAS {expected}. Se detectaron: {df.columns.tolist()}")
        return None
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _read_outs_csv_path(path: Path) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = _read_csv_robusto_from_path(path)
    if df is None:
        return None
    expected = ["Out1", "Out2", "Out3", "Velocidad", "Ra_bd"]
    df.columns = [c.strip() for c in df.columns]
    if set(df.columns) == set(expected):
        df = df[expected]
    elif df.columns.tolist() != expected:
        st.sidebar.error(f"OUTS inválido. Columnas esperadas {expected}. Se detectaron: {df.columns.tolist()}")
        return None
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if outs_csv_up is not None:
    df_outs_loaded = _read_outs_csv_filelike(outs_csv_up)
    outs_source_label = "Uploader"
else:
    df_outs_loaded = _read_outs_csv_path(auto_outs_path) if auto_outs_path else None
    outs_source_label = f"Carga automática: {auto_outs_path.name}" if auto_outs_path else "Sin OUTS detectado"

if df_outs_loaded is None or df_outs_loaded.empty:
    st.sidebar.warning("OUTS no disponible o inválido (se puede continuar sin clasificación por rangos).")
else:
    st.sidebar.success(f"OUTS cargado ({outs_source_label})")
    with st.sidebar.expander("Vista rápida OUTS", expanded=False):
        st.dataframe(df_outs_loaded.head(20), use_container_width=True)

# ---- Cargar IMAGEN ----
st.sidebar.header("Imagen")
up = st.sidebar.file_uploader("png/jpg/tif/bmp", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if up is None:
    st.info("Sube una imagen y coloca la ROI para empezar.")
    st.stop()

# --- leer imagen ---
raw_bytes = up.read()
I0 = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
if I0 is None:
    st.error("No se pudo leer la imagen.")
    st.stop()
base_bgr = I0 if I0.ndim == 3 else cv2.cvtColor(I0, cv2.COLOR_GRAY2BGR)

sat_pct = detect_overexposed(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY))
if sat_pct > 2.0:
    st.warning(f"⚠️ Imagen potencialmente sobreexpuesta: {sat_pct:.1f}% de píxeles cerca de saturación.")

# Vista previa para orientar el canvas (sin mostrar ROI aún)
preview_bgr = cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

Igray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
Igray = equalize_adapthist(np.clip(img_as_float(Igray), 0, 1), kernel_size=(8, 8), clip_limit=0.01).astype(np.float32)

# ========================= STATE KEYS =========================
if "ran" not in st.session_state:
    st.session_state["ran"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

# ========================= Función de proceso =========================
def ejecutar_pipeline(Igray_eq: np.ndarray,
                      material_sel: str,
                      vel_sel,
                      df_norm_for_ui: pd.DataFrame | None,
                      REF_DB: pd.DataFrame,
                      df_outs_loaded: pd.DataFrame | None,
                      roi_box_imgcoords: tuple[int, int, int, int]) -> dict:
    """Corre filtros, OUTs y predicción; devuelve diccionario con todos los resultados."""
    x0, y0, x1, y1 = roi_box_imgcoords
    Irect = Igray_eq[y0:y1, x0:x1]

    # Filtros y OUTs
    Iproc_f32 = aplicar_filtros_cap7(Irect)
    Iproc_u8 = to_uint8_display(Iproc_f32)
    hist, out1, out2, out3 = analisis_histograma(Iproc_u8, ref_area_px=OUT1_STD_AREA_PX, smooth=1)

    # Referencia BD (µm)
    Ra_bd_um = np.nan
    if vel_sel is not None:
        if (df_norm_for_ui is not None) and (not df_norm_for_ui.empty):
            df_sel_ref = df_norm_for_ui[
                (df_norm_for_ui["Material"] == material_sel) &
                (df_norm_for_ui["Velocidad"] == vel_sel)
            ]
        else:
            df_sel_ref = REF_DB[
                (REF_DB["Material"] == material_sel) &
                (REF_DB["Velocidad"] == vel_sel)
            ]
        if (df_sel_ref is not None) and (not df_sel_ref.empty):
            Ra_bd_um = float(df_sel_ref["Rugosidad"].mean())

    # Estimación por OUTs
    Ra_outs_um = np.nan
    metodo_outs = "—"
    rangos_df = None
    centroides_df = None
    vecinos_df = None
    spread_w = np.nan
    if df_outs_loaded is not None and not df_outs_loaded.empty:
        rangos_df, centroides_df = compute_rangos_y_centroides_from_outs(df_outs_loaded, q_low=0.05, q_high=0.95)
        Ra_outs_um, info_pred = clasificar_o_predecir_ra(
            out1=float(out1), out2=float(out2), out3=float(out3),
            df_outs=df_outs_loaded, rangos_df=rangos_df, centroides_df=centroides_df
        )
        metodo_outs = info_pred.get("metodo", "outs")
        vecinos_df = info_pred.get("vecinos", None)
        spread_w = info_pred.get("spread_w", np.nan)

    # Error relativo
    err_outs_pct = (
        abs(Ra_outs_um - Ra_bd_um) / max(Ra_bd_um, 1e-9) * 100.0
    ) if (np.isfinite(Ra_outs_um) and np.isfinite(Ra_bd_um)) else np.nan

    return {
        "roi_img": Irect,
        "Iproc_u8": Iproc_u8,
        "hist": hist,
        "out1": int(out1),
        "out2": int(out2),
        "out3": float(out3),
        "Ra_bd_um": float(Ra_bd_um) if np.isfinite(Ra_bd_um) else np.nan,
        "Ra_outs_um": float(Ra_outs_um) if np.isfinite(Ra_outs_um) else np.nan,
        "metodo_outs": metodo_outs,
        "vecinos_df": vecinos_df,
        "rangos_df": rangos_df,
        "centroides_df": centroides_df,
        "spread_w": float(spread_w) if np.isfinite(spread_w) else np.nan,
        "err_outs_pct": float(err_outs_pct) if np.isfinite(err_outs_pct) else np.nan,
    }

# ========================= Tabs =========================
tab1, tab2 = st.tabs(["ROI", "Procesar y Resultados"])

# ---- ROI (RECTÁNGULO FIJO 2314 × 443, ARRASTRABLE) + botón Run ----
with tab1:
    st.subheader("Posiciona la ROI fija arrastrando")

    h_img, w_img = preview_bgr.shape[:2]
    s_rect = compute_display_scale(w_img, h_img, MAX_CANVAS_W, MAX_CANVAS_H)
    bg_pil = _to_pil_rgb(preview_bgr)
    if s_rect < 1.0:
        bg_pil = bg_pil.resize((int(round(w_img * s_rect)), int(round(h_img * s_rect))), resample=PILImage.BILINEAR)

    H_img, W_img = Igray.shape[:2]

    # Estado inicial ROI en coordenadas de IMAGEN
    if "roi_xy" not in st.session_state:
        x0_def = max(0, (W_img - ROI_W_PX) // 2)
        y0_def = max(0, (H_img - ROI_H_PX) // 2)
        st.session_state["roi_xy"] = (x0_def, y0_def)

    # Debounce timestamp de actualizaciones
    if "roi_last_ts" not in st.session_state:
        st.session_state["roi_last_ts"] = 0.0

    # Convertir ROI imagen -> canvas para dibujar
    x0_img, y0_img = st.session_state["roi_xy"]
    x0_can = int(round(x0_img * s_rect))
    y0_can = int(round(y0_img * s_rect))
    w_can = int(round(ROI_W_PX * s_rect))
    h_can = int(round(ROI_H_PX * s_rect))

    # Guardar/recuperar el objeto del canvas para evitar que se re-cree en cada rerun
    if "roi_can_obj" not in st.session_state:
        st.session_state["roi_can_obj"] = _make_fixed_rect(x0_can, y0_can, w_can, h_can)

    # Si cambió la escala, ajusta ancho/alto manteniendo left/top actuales
    obj_can = st.session_state["roi_can_obj"]
    obj_can["width"], obj_can["height"] = w_can, h_can

    tr_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=bg_pil,
        height=bg_pil.size[1],
        width=bg_pil.size[0],
        drawing_mode="transform",
        initial_drawing={"objects": [obj_can]},   # usa SIEMPRE el último objeto persistido
        key="canvas_roi_drag_transform",
        update_streamlit=True,
    )

    # Lee cambios del canvas y actualiza estado con pequeño debounce
    if tr_canvas.json_data and "objects" in tr_canvas.json_data and tr_canvas.json_data["objects"]:
        obj_new = tr_canvas.json_data["objects"][0]
        left_can_new = int(round(obj_new.get("left", obj_can.get("left", x0_can))))
        top_can_new  = int(round(obj_new.get("top",  obj_can.get("top",  y0_can))))

        # Solo si realmente cambió la posición
        if (left_can_new != obj_can.get("left")) or (top_can_new != obj_can.get("top")):
            now = time.time()
            # Debounce ~120 ms
            if (now - st.session_state["roi_last_ts"]) >= 0.12:
                # Actualiza objeto persistido (canvas)
                obj_can["left"] = left_can_new
                obj_can["top"]  = top_can_new
                st.session_state["roi_can_obj"] = obj_can

                # Actualiza ROI en coordenadas de imagen (con límites)
                x0_img_new = int(round(left_can_new / max(s_rect, 1e-9)))
                y0_img_new = int(round(top_can_new  / max(s_rect, 1e-9)))
                x0_img_new = max(0, min(x0_img_new, W_img - ROI_W_PX))
                y0_img_new = max(0, min(y0_img_new, H_img - ROI_H_PX))
                st.session_state["roi_xy"] = (x0_img_new, y0_img_new)

                st.session_state["roi_last_ts"] = now

    x0, y0 = st.session_state["roi_xy"]
    x1, y1 = x0 + ROI_W_PX, y0 + ROI_H_PX
    st.session_state["rect_coords"] = (x0, y0, x1, y1)

    # NOTA: NO mostramos la previsualización de la ROI aquí.
    st.info(f"ROI actual lista para ejecutar: x0={x0}, y0={y0}, ancho={ROI_W_PX}, alto={ROI_H_PX} (px)")

    # --------- Botón Run (dispara todo y navega a Resultados) ---------
    if st.button("▶️ Run", type="primary", use_container_width=True):
        roi_box = st.session_state["rect_coords"]
        res = ejecutar_pipeline(
            Igray_eq=Igray,
            material_sel=material_sel,
            vel_sel=vel_sel,
            df_norm_for_ui=df_norm_for_ui,
            REF_DB=REF_DB,
            df_outs_loaded=df_outs_loaded,
            roi_box_imgcoords=roi_box
        )
        st.session_state["results"] = res
        st.session_state["ran"] = True

        # Navegar automáticamente a la pestaña "Procesar y Resultados"
        components.html("""
            <script>
            const attempt = () => {
              const btns = Array.from(parent.document.querySelectorAll('button'));
              const target = btns.find(b => b.innerText.trim() === 'Procesar y Resultados');
              if (target) { target.click(); }
              else { setTimeout(attempt, 100); }
            };
            attempt();
            </script>
        """, height=0)

# ========================= Procesar / Resultados =========================
with tab2:
    st.subheader("Procesar / Resultados")

    if not st.session_state.get("ran", False) or st.session_state.get("results") is None:
        st.info("Configura la ROI y presiona **Run** en la pestaña *ROI* para ver resultados aquí.")
    else:
        res = st.session_state["results"]

        roi_vis = res["roi_img"]
        # Si llega vertical (alto > ancho), rotamos solo para la VISUALIZACIÓN
        if roi_vis.ndim >= 2 and roi_vis.shape[0] > roi_vis.shape[1]:
            roi_vis = cv2.rotate(roi_vis, cv2.ROTATE_90_CLOCKWISE)

        col_img, col_spacer = st.columns([1, 1])
        with col_img:
            st.image(
                roi_vis,
                caption=f"ROI seleccionada {roi_vis.shape[1]}×{roi_vis.shape[0]} px (vista horizontal)",
                clamp=True,
                use_column_width=False,
                width=700,
            )


        # 2) Rugosidades y errores
        st.subheader("Rugosidades y errores")
        if df_outs_loaded is None or df_outs_loaded.empty:
            st.warning("Carga un CSV de OUTS válido para estimar por OUTs.")
            st.metric("Ra Base de datos (µm)", f"{res['Ra_bd_um']:.2f}" if np.isfinite(res['Ra_bd_um']) else "—")
        else:
            cA, cB, cC = st.columns(3)
            cA.metric(
                "Ra por OUTs",
                f"{res['Ra_outs_um']:.2f} µm" if np.isfinite(res['Ra_outs_um']) else "—",
                help=f"Método: {res['metodo_outs']}"
            )
            cB.metric(
                "Ra BD",
                f"{res['Ra_bd_um']:.2f} µm" if np.isfinite(res['Ra_bd_um']) else "—"
            )
            cC.metric(
                "Error OUTs vs BD",
                f"{res['err_outs_pct']:.1f} %" if np.isfinite(res['err_outs_pct']) else "—"
            )

            if isinstance(res.get("vecinos_df"), pd.DataFrame) and not res["vecinos_df"].empty:
                with st.expander("Vecinos kNN usados (Ra_vecino, dist, peso)", expanded=False):
                    st.dataframe(res["vecinos_df"], use_container_width=True, height=220)

            if isinstance(res.get("rangos_df"), pd.DataFrame) and not res["rangos_df"].empty:
                with st.expander("Rangos 5–95% (desde OUTS CSV)", expanded=False):
                    st.dataframe(res["rangos_df"], use_container_width=True, height=220)
            if isinstance(res.get("centroides_df"), pd.DataFrame) and not res["centroides_df"].empty:
                with st.expander("Centroides (medianas OUT por Ra_bd)", expanded=False):
                    st.dataframe(res["centroides_df"], use_container_width=True, height=220)

        st.divider()

        # 3) OUTs calculados y promedio de BD
        st.subheader("Parámetros OUTs")

        # OUTs calculados
        c1, c2, c3 = st.columns(3)
        c1.metric("Out1", f"{res['out1']} px")
        c2.metric("Out2", f"{res['out2']}")
        c3.metric("Out3", f"{res['out3']:.2f}")

        # OUTs promedio de la BD para la velocidad seleccionada
        if df_outs_loaded is not None and not df_outs_loaded.empty and vel_sel is not None:
            df_outs_vel = df_outs_loaded[df_outs_loaded["Velocidad"] == vel_sel]
            if not df_outs_vel.empty:
                out1_avg = df_outs_vel["Out1"].mean()
                out2_avg = df_outs_vel["Out2"].mean()
                out3_avg = df_outs_vel["Out3"].mean()

                st.markdown("---")
                st.markdown(f"**Promedios de la BD para Velocidad {vel_sel} mm/min:**")
                cA, cB, cC = st.columns(3)
                cA.metric("Out1 promedio", f"{out1_avg:.0f} px")
                cB.metric("Out2 promedio", f"{out2_avg:.0f}")
                cC.metric("Out3 promedio", f"{out3_avg:.2f}")
            else:
                st.info(f"No hay registros de OUTs en la BD para Velocidad {vel_sel}.")
        else:
            st.warning(
                "No se pudo calcular el promedio de OUTs — carga un CSV de OUTS válido y selecciona una velocidad.")

        st.divider()

        # 4) Histograma normalizado
        st.subheader("Histograma normalizado")
        fig_h, axh = plt.subplots()
        axh.plot(res["hist"], 'k', lw=1)
        axh.set_title(f"Histograma normalizado (ref={OUT1_STD_AREA_PX:,} px)")
        axh.set_xlim(0, 255); axh.grid(True, alpha=.3)
        st.pyplot(fig_h)
