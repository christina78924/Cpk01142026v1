import streamlit as st
import pandas as pd
import numpy as np
import io

# =========================
# 全域設定
# =========================
st.set_page_config(page_title="IPQC CPK & Yield Generator", layout="wide")
st.title("📊 IPQC CPK & Yield 報表生成器")

CPK_LIMIT = 1.33

# =========================
# 站點順序
# =========================
TARGET_ORDER = [
    "MLA assy installation",
    "Mirror attachment",
    "Barrel attachment",
    "Condenser lens attach",
    "LED Module  attachment",
    "ILLU Module cover attachment",
    "Relay lens attachment",
    "LED FLEX GRAPHITE-1",
    "reflector attach",
    "singlet attach",
    "HWP Mylar attach",
    "PBS attachment",
    "Doublet attachment",
    "Top cover installation",
    "PANEL PRECISION AA（LAA）",
    "POST DAA INSPECTION",
    "PANEL FLEX ASSY",
    "LCOS GRAPHITE ATTACH",
    "DE OQC"
]

# =========================
# 工具函式
# =========================
def normalize_name(name):
    return (
        name.lower()
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("（", "")
        .replace("）", "")
        .replace("-", "")
        .replace("_", "")
    )

TARGET_MAP = {normalize_name(n): n for n in TARGET_ORDER}

def calculate_cpk(data, usl, lsl):
    data = pd.to_numeric(data, errors="coerce").dropna()
    if len(data) < 2:
        return np.nan
    mean = data.mean()
    std = data.std(ddof=1)
    if std == 0:
        return np.nan
    cpu = (usl - mean) / (3 * std) if not pd.isna(usl) else np.nan
    cpl = (mean - lsl) / (3 * std) if not pd.isna(lsl) else np.nan
    return min(cpu, cpl) if not pd.isna(cpu) and not pd.isna(cpl) else cpu if not pd.isna(cpu) else cpl

def excel_col_letter(idx):
    letters = ""
    while idx >= 0:
        idx, r = divmod(idx, 26)
        letters = chr(65 + r) + letters
        idx -= 1
    return letters

def highlight_low_cpk(val):
    try:
        if float(val) < CPK_LIMIT:
            return "background-color:#ffc7ce;color:#9c0006;font-weight:bold"
    except:
        pass
    return ""

# =========================
# Yield (Overall per Station)
# =========================
def process_yield(station, df):
    best_col, max_cnt = -1, 0
    for c in range(min(30, df.shape[1])):
        col = df.iloc[:, c].astype(str).str.upper()
        total = (col == "OK").sum() + (col == "NG").sum()
        if total > max_cnt:
            max_cnt, best_col = total, c
    if best_col == -1:
        return None
    col = df.iloc[:, best_col].astype(str).str.upper()
    ok = (col == "OK").sum()
    ng = (col == "NG").sum()
    return {
        "Station": station,
        "Total Qty": ok + ng,
        "OK Qty": ok,
        "NG Qty": ng,
        "Yield": ok / (ok + ng) if (ok + ng) else 0
    }

# =========================
# Yield (By Config) - 新增
# =========================
def process_yield_by_config(station, df):
    # 尋找 OK/NG 結果欄位
    best_col, max_cnt = -1, 0
    for c in range(min(30, df.shape[1])):
        col = df.iloc[:, c].astype(str).str.upper()
        total = (col == "OK").sum() + (col == "NG").sum()
        if total > max_cnt:
            max_cnt, best_col = total, c
            
    if best_col == -1:
        return []

    # Config 欄位固定在 index 1
    config_col_idx = 1
    if df.shape[1] <= config_col_idx:
        return []
        
    # 提取 Config 與 Result 欄位
    # 直接使用 iloc 提取並建立副本
    subset = df.iloc[:, [config_col_idx, best_col]].copy()
    subset.columns = ["Config", "Result"]
    
    subset["Result"] = subset["Result"].astype(str).str.upper()
    subset["Config"] = subset["Config"].astype(str).fillna("")
    
    # 篩選出有效的判定結果
    subset = subset[subset["Result"].isin(["OK", "NG"])]
    
    if subset.empty:
        return []
        
    results = []
    for config, g in subset.groupby("Config"):
        ok = (g["Result"] == "OK").sum()
        ng = (g["Result"] == "NG").sum()
        results.append({
            "Station": station,
            "Config": config,
            "OK": ok,
            "NG": ng
        })
    return results

# =========================
# 共用資料提取邏輯
# =========================
def _extract_data_for_cpk(df):
    def find_row(keywords):
        for i in range(min(60, len(df))):
            row = " ".join(df.iloc[i].astype(str).str.lower())
            if any(k in row for k in keywords):
                return i
        return -1

    dim_row = find_row(["dim. no", "dim no"])
    usl_row = find_row(["usl"])
    lsl_row = find_row(["lsl"])
    config_header_row = find_row(["config"])

    if dim_row == -1:
        return None, None, None, None

    headers = df.iloc[dim_row].astype(str)
    ignore = {"date", "time", "no.", "remark", "judge", "note", "nan", ""}
    dim_cols = {i: h for i, h in enumerate(headers) if h.lower().strip() not in ignore and len(h.strip()) > 1}

    usls, lsls = {}, {}
    if usl_row != -1:
        for i, v in enumerate(df.iloc[usl_row]):
            try: usls[i] = float(v)
            except: pass
    if lsl_row != -1:
        for i, v in enumerate(df.iloc[lsl_row]):
            try: lsls[i] = float(v)
            except: pass

    start_row_candidates = [dim_row, usl_row, lsl_row]
    if config_header_row != -1:
        start_row_candidates.append(config_header_row)
    
    start = max(start_row_candidates) + 1
    data = df.iloc[start:].copy()

    date_col = -1
    for c in range(min(15, data.shape[1])):
        if data.iloc[:, c].astype(str).str.contains(r"202\d-\d{2}-\d{2}").any():
            date_col = c
            break
    if date_col == -1:
        return None, None, None, None

    data["Date"] = data.iloc[:, date_col].astype(str).str.extract(r"(202\d-\d{2}-\d{2})")[0]
    data = data.dropna(subset=["Date"])

    config_col_idx = 1
    if data.shape[1] > config_col_idx:
        data["Config"] = data.iloc[:, config_col_idx].astype(str).fillna("")
    else:
        data["Config"] = ""

    return data, dim_cols, usls, lsls

# =========================
# CPK Functions
# =========================
def process_cpk(station, df):
    data, dim_cols, usls, lsls = _extract_data_for_cpk(df)
    if data is None: 
        return []

    results = []
    for (date, config_val), g in data.groupby(["Date", "Config"]):
        for idx, dim in dim_cols.items():
            vals = g.iloc[:, idx]
            cpk = calculate_cpk(vals, usls.get(idx), lsls.get(idx))
            n = pd.to_numeric(vals, errors="coerce").dropna().size
            if n > 0:
                results.append({
                    "Station": station,
                    "Dim No": dim,
                    "Config": config_val,
                    "Date": date,
                    "Sample Size": n,
                    "USL": usls.get(idx, ""),
                    "LSL": lsls.get(idx, ""),
                    "CPK": round(cpk, 3) if not pd.isna(cpk) else ""
                })
    return results

def process_cpk_by_config(station, df):
    data, dim_cols, usls, lsls = _extract_data_for_cpk(df)
    if data is None: 
        return []

    results = []
    for config_val, g in data.groupby("Config"):
        for idx, dim in dim_cols.items():
            vals = g.iloc[:, idx]
            cpk = calculate_cpk(vals, usls.get(idx), lsls.get(idx))
            n = pd.to_numeric(vals, errors="coerce").dropna().size
            if n > 0:
                results.append({
                    "Station": station,
                    "Dim No": dim,
                    "Config": config_val,
                    "Sample Size": n,
                    "USL": usls.get(idx, ""),
                    "LSL": lsls.get(idx, ""),
                    "CPK": round(cpk, 3) if not pd.isna(cpk) else ""
                })
    return results

# =========================
# Streamlit 主流程
# =========================
uploaded = st.file_uploader("📂 上傳 Excel (.xlsx)", type=["xlsx"])

if uploaded:
    xls = pd.ExcelFile(uploaded)
    yield_list, cpk_list, cpk_config_list = [], [], []
    yield_config_counts = [] # 收集 raw counts

    for sheet in xls.sheet_names:
        norm = normalize_name(sheet)
        station = next((v for k, v in TARGET_MAP.items() if k in norm), None)
        if not station:
            continue
        df = pd.read_excel(uploaded, sheet_name=sheet, header=None)
        
        # 1. Yield (Overall)
        y = process_yield(station, df)
        if y:
            yield_list.append(y)
            
        # 2. Yield (By Config) - 收集數據
        yield_config_counts.extend(process_yield_by_config(station, df))
        
        # 3. CPK Processes
        cpk_list.extend(process_cpk(station, df))
        cpk_config_list.extend(process_cpk_by_config(station, df))

    # 建立 DataFrames
    df_yield = pd.DataFrame(yield_list)
    df_cpk = pd.DataFrame(cpk_list)
    df_cpk_config = pd.DataFrame(cpk_config_list)
    
    # 處理 Yield by Config 匯總 (因為可能跨 Sheet)
    df_yield_agg = pd.DataFrame()
    if yield_config_counts:
        df_counts = pd.DataFrame(yield_config_counts)
        # 依照 Station, Config 加總 OK 與 NG
        df_yield_agg = df_counts.groupby(["Station", "Config"], as_index=False)[["OK", "NG"]].sum()
        df_yield_agg["Total"] = df_yield_agg["OK"] + df_yield_agg["NG"]
        # 計算良率 (浮點數)
        df_yield_agg["Yield Raw"] = df_yield_agg.apply(
            lambda r: r["OK"] / r["Total"] if r["Total"] > 0 else 0, axis=1
        )

    # 產生 CPK Summary
    df_summary = pd.DataFrame()
    if not df_cpk_config.empty:
        df_for_stats = df_cpk_config.copy()
        df_for_stats['CPK'] = pd.to_numeric(df_for_stats['CPK'], errors='coerce')
        df_for_stats = df_for_stats.dropna(subset=['CPK'])
        
        if not df_for_stats.empty:
            # 計算 Min/Max
            stats = df_for_stats.groupby(['Station', 'Config'])['CPK'].agg(['min', 'max']).reset_index()
            # 找出 Min 對應的 Dim No
            loc_min = df_for_stats.groupby(['Station', 'Config'])['CPK'].idxmin()
            min_dims = df_for_stats.loc[loc_min, ['Station', 'Config', 'Dim No']].rename(columns={'Dim No': 'Min CPK Dim'})
            
            # 合併統計數據
            df_summary = pd.merge(stats, min_dims, on=['Station', 'Config'], how='left')
            df_summary.columns = ['Station', 'Config', 'Min CPK', 'Max CPK', 'Min CPK Dim']
            
            # [新增] 合併 Yield by Config 資訊
            if not df_yield_agg.empty:
                df_summary = pd.merge(df_summary, df_yield_agg[["Station", "Config", "Yield Raw"]], 
                                    on=["Station", "Config"], how="left")
                # 格式化良率
                df_summary["Yield"] = df_summary["Yield Raw"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
            else:
                df_summary["Yield"] = ""

            # 調整欄位順序: Station, Config, Yield, Min CPK Dim, Min CPK, Max CPK
            cols = ['Station', 'Config', 'Yield', 'Min CPK Dim', 'Min CPK', 'Max CPK']
            # 防呆：確保欄位存在才選取
            cols = [c for c in cols if c in df_summary.columns]
            df_summary = df_summary[cols]
            
            # 排序
            df_summary["Station"] = pd.Categorical(df_summary["Station"], TARGET_ORDER, ordered=True)
            df_summary = df_summary.sort_values(['Station', 'Config'])

    # 顯示處理 (Sorting & Formatting for other tabs)
    if not df_yield.empty:
        df_yield["Station"] = pd.Categorical(df_yield["Station"], TARGET_ORDER, ordered=True)
        df_yield = df_yield.sort_values("Station")
        df_yield["Yield"] = (df_yield["Yield"] * 100).round(2).astype(str) + "%"

    if not df_cpk.empty:
        df_cpk["Station"] = pd.Categorical(df_cpk["Station"], TARGET_ORDER, ordered=True)
        df_cpk = df_cpk.sort_values(["Station", "Dim No", "Date", "Config"])
        
    if not df_cpk_config.empty:
        df_cpk_config["Station"] = pd.Categorical(df_cpk_config["Station"], TARGET_ORDER, ordered=True)
        df_cpk_config = df_cpk_config.sort_values(["Station", "Dim No", "Config"])

    # 建立分頁
    tab1, tab2, tab3, tab4 = st.tabs(["Yield", "CPK", "CPK by Config", "CPK Summary"])

    with tab1:
        st.dataframe(df_yield, use_container_width=True)
    with tab2:
        st.write(df_cpk.style.applymap(highlight_low_cpk, subset=["CPK"]))
    with tab3:
        st.write(df_cpk_config.style.applymap(highlight_low_cpk, subset=["CPK"]))
    with tab4:
        st.write("統計每個站點與 Config 下的 Yield 及 CPK 極值")
        # 僅對 CPK 數值欄位進行紅色警示
        st.write(df_summary.style.applymap(highlight_low_cpk, subset=["Min CPK", "Max CPK"]))

    # ===== Excel 匯出 =====
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_yield.to_excel(writer, sheet_name="Yield", index=False)
        df_cpk.to_excel(writer, sheet_name="CPK", index=False)
        df_cpk_config.to_excel(writer, sheet_name="CPK by Config", index=False)
        df_summary.to_excel(writer, sheet_name="CPK Summary", index=False)

        # 格式化設定
        wb = writer.book
        red = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006", "bold": True})
        
        def apply_cond_format(sheet_name, df, cols_to_check):
            if df.empty: return
            ws = writer.sheets[sheet_name]
            for col_name in cols_to_check:
                if col_name in df.columns:
                    col_letter = excel_col_letter(df.columns.get_loc(col_name))
                    ws.conditional_format(
                        f"{col_letter}2:{col_letter}{len(df)+1}",
                        {"type": "cell", "criteria": "<", "value": CPK_LIMIT, "format": red}
                    )

        apply_cond_format("CPK", df_cpk, ["CPK"])
        apply_cond_format("CPK by Config", df_cpk_config, ["CPK"])
        apply_cond_format("CPK Summary", df_summary, ["Min CPK", "Max CPK"])

    output.seek(0)
    st.download_button(
        "📥 下載 Excel",
        output,
        "IPQC_Report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )