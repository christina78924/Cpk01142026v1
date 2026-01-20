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
# 站點順序 (Station Order)
# =========================
# [修正] 將 PANEL FLEX ASSY 系列移至 LCOS GRAPHITE ATTACH 之後
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
    "LCOS GRAPHITE ATTACH",   # 先
    "PANEL FLEX ASSY",        # 後 (原本在前面)
    "PANEL FLEX ASSY-2",      # 後 (新增)
    "DE OQC"
]

# =========================
# 分類規則設定 (Category Rules)
# =========================
CATEGORY_RULES_RAW = {
    "MLA assy installation": {
        "Category 1": ["CPK201-new", "CPK202", "CPK203"],
        "Category 2": ["CPK204", "CPK205-X", "CPK205-Y"]
    },
    "Mirror attachment": {
        "Category 1": ["CPK305"],
        "Category 2": ["CPK301", "CPK302", "CPK303"]
    },
    "Barrel attachment": {
        "Category 1": ["CPK206", "CPK207", "CPK208"],
        "Category 2": ["CPK201", "CPK203", "CPK204", "CPK205"]
    },
    "Condenser lens attach": {
        "Category 1": ["CPK403"]
    },
    "LED Module  attachment": {
        "Category 1": ["CPK403"]
    },
    "ILLU Module cover attachment": {
        "Category 1": ["CPK601", "CPK602"]
    },
    "Relay lens attachment": {
        "Category 1": ["CPK701", "CPK702", "CPK703"],
        "Category 2": ["CPK704-1", "CPK704-2", "CPK704-3", "CPK704-4"]
    },
    "reflector attach": {
        "Category 1": ["CPK203", "CPK204", "CPK205"]
    },
    "singlet attach": {
        "Category 1": ["CPK202", "CPK225", "CPK256"]
    },
    "HWP Mylar attach": {
        "Category 1": ["CPK218", "CPK220", "CPK226"]
    },
    "PBS attachment": {
        "Category 1": ["CPK205__#2", "CPK203__#2", "CPK204-Tilt__#2"],
        "Category 2": ["CPK202", "CPK201-rotation", "TIP"]
    },
    "Doublet attachment": {
        "Category 1": ["CPK301-1__#2", "CPK301-X__#2", "CPK301-Y__#2"],
        "Category 2": ["CPK303"]
    },
    "Top cover installation": {
        "Category 1": ["CPK401-T1", "CPK401-T2", "CPK401-T3", "CPK401-T4"]
    },
    "PANEL PRECISION AA（LAA）": {
        "Category 1": ["CPK503", "CPK504", "CPK505", "CPK610", "CPK611", "CPK612"],
        "Category 2": ["CPK604", "CPK605__#2", "CPK607__#2", "CPK608", "CPK609__#2"]
    },
    "LED FLEX GRAPHITE-1": {
        "Category 1": ["Cpk801"],
        "Category 2": ["Cpk802"]
    },
    "POST DAA INSPECTION": {
        "Category 1": ["Cpk201", "Cpk220", "Cpk203"],
        "Category 2": ["CPK227/FAI204-T1", "CPK227/FAI204-T2", "CPK227/FAI204-T3", "CPK227/FAI204-T4", "Cpk209", "CPK229-1", "CPK229-2"]
    },
    "LCOS GRAPHITE ATTACH": {
        "Category 1": ["Cpk214"]
    },
    # [新增]
    "PANEL FLEX ASSY": {
        "Category 1": ["ALL"]
    },
    # [新增]
    "PANEL FLEX ASSY-2": {
        "Category 1": ["ALL"]
    },
    "DE OQC": {
        "Category 1": ["ALL"]
    }
}

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

# 預處理分類規則
CATEGORY_CONFIG = {}
for station, cats in CATEGORY_RULES_RAW.items():
    norm_stat = normalize_name(station)
    CATEGORY_CONFIG[norm_stat] = {}
    for cat_name, dim_list in cats.items():
        if dim_list == ["ALL"]:
            CATEGORY_CONFIG[norm_stat][cat_name] = "ALL"
        else:
            CATEGORY_CONFIG[norm_stat][cat_name] = [d.lower().strip() for d in dim_list]

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

# 顏色邏輯 (紅 < 0.67, 橘 < 1.0, 黃 < 1.33)
def highlight_low_cpk(val):
    try:
        v = float(val)
        if v < 0.67:
            return "background-color:#ffc7ce;color:#9c0006;font-weight:bold" # 紅色
        elif v < 1.0:
            return "background-color:#ffeb9c;color:#9c5700;font-weight:bold" # 橘色
        elif v < 1.33:
            return "background-color:#ffffcc;color:#808000;font-weight:bold" # 黃色
    except:
        pass
    return ""

# =========================
# Yield Logic
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

def process_yield_by_config(station, df):
    best_col, max_cnt = -1, 0
    for c in range(min(30, df.shape[1])):
        col = df.iloc[:, c].astype(str).str.upper()
        total = (col == "OK").sum() + (col == "NG").sum()
        if total > max_cnt:
            max_cnt, best_col = total, c
    if best_col == -1:
        return []

    config_col_idx = 1
    if df.shape[1] <= config_col_idx:
        return []
        
    subset = df.iloc[:, [config_col_idx, best_col]].copy()
    subset.columns = ["Config", "Result"]
    subset["Result"] = subset["Result"].astype(str).str.upper()
    subset["Config"] = subset["Config"].astype(str).fillna("")
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
# CPK Extraction Logic
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

    raw_headers = df.iloc[dim_row].astype(str)
    header_counts = {}
    unique_headers = []
    
    for h in raw_headers:
        h_clean = h.strip()
        key = h_clean.lower()
        if key in header_counts:
            header_counts[key] += 1
            unique_name = f"{h_clean}__#{header_counts[key]}" 
        else:
            header_counts[key] = 1
            unique_name = h_clean
        unique_headers.append(unique_name)
    
    headers = pd.Series(unique_headers)
    
    ignore = {"date", "time", "no.", "remark", "judge", "note", "nan", ""}
    dim_cols = {}
    for i, h in enumerate(headers):
        base_name = h.split("__#")[0].lower()
        if base_name not in ignore and len(base_name) > 1:
            dim_cols[i] = h

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
# CPK Processing
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
    yield_config_counts = []

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
            
        # 2. Yield (By Config)
        yield_config_counts.extend(process_yield_by_config(station, df))
        
        # 3. CPK Processes
        cpk_list.extend(process_cpk(station, df))
        cpk_config_list.extend(process_cpk_by_config(station, df))

    # DataFrames
    df_yield = pd.DataFrame(yield_list)
    df_cpk = pd.DataFrame(cpk_list)
    df_cpk_config = pd.DataFrame(cpk_config_list)
    
    # Process Yield by Config
    df_yield_agg = pd.DataFrame()
    if yield_config_counts:
        df_counts = pd.DataFrame(yield_config_counts)
        df_yield_agg = df_counts.groupby(["Station", "Config"], as_index=False)[["OK", "NG"]].sum()
        df_yield_agg["Total"] = df_yield_agg["OK"] + df_yield_agg["NG"]
        df_yield_agg["Yield Raw"] = df_yield_agg.apply(
            lambda r: r["OK"] / r["Total"] if r["Total"] > 0 else 0, axis=1
        )

    # =========================
    # Summary & Analysis & Categorized
    # =========================
    df_summary = pd.DataFrame()
    df_analysis = pd.DataFrame()
    df_categorized_wide = pd.DataFrame()

    if not df_cpk_config.empty:
        df_for_stats = df_cpk_config.copy()
        df_for_stats['CPK'] = pd.to_numeric(df_for_stats['CPK'], errors='coerce')
        df_for_stats = df_for_stats.dropna(subset=['CPK'])
        
        if not df_for_stats.empty:
            # --- 1. CPK Summary & Analysis ---
            stats = df_for_stats.groupby(['Station', 'Config'])['CPK'].agg(['min', 'max']).reset_index()
            loc_min = df_for_stats.groupby(['Station', 'Config'])['CPK'].idxmin()
            min_dims = df_for_stats.loc[loc_min, ['Station', 'Config', 'Dim No']].rename(columns={'Dim No': 'Min CPK Dim'})
            
            df_summary = pd.merge(stats, min_dims, on=['Station', 'Config'], how='left')
            df_summary.columns = ['Station', 'Config', 'Min CPK', 'Max CPK', 'Min CPK Dim']
            
            def get_failed_dims(g):
                dims_133 = g[g['CPK'] < 1.33]['Dim No'].unique()
                dims_100 = g[g['CPK'] < 1.0]['Dim No'].unique()
                return pd.Series({
                    'Dims CPK < 1.33': ", ".join(map(str, dims_133)),
                    'Dims CPK < 1.0': ", ".join(map(str, dims_100))
                })

            failed_lists = df_for_stats.groupby(['Station', 'Config']).apply(get_failed_dims).reset_index()
            
            # Merge Yield
            if not df_yield_agg.empty:
                yield_info = df_yield_agg[["Station", "Config", "Yield Raw", "Total"]]
                df_summary = pd.merge(df_summary, yield_info, on=["Station", "Config"], how="left")
            else:
                df_summary["Yield Raw"] = np.nan
                df_summary["Total"] = 0

            df_summary["Yield"] = df_summary["Yield Raw"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
            df_summary["Sample Size"] = df_summary["Total"].fillna(0).astype(int)

            df_analysis = pd.merge(df_summary, failed_lists, on=['Station', 'Config'], how='left')
            
            # --- 2. CPK Categorized Analysis (Wide Format) ---
            def identify_category(row):
                station_norm = normalize_name(row['Station'])
                dim_norm = row['Dim No'].lower().strip()
                rules = CATEGORY_CONFIG.get(station_norm, {})
                for cat, dim_list in rules.items():
                    if dim_list == "ALL":
                        return cat
                    if dim_norm in dim_list:
                        return cat
                return None

            df_cat_source = df_for_stats.copy()
            df_cat_source['Category'] = df_cat_source.apply(identify_category, axis=1)
            df_cat_source = df_cat_source.dropna(subset=['Category'])
            
            if not df_cat_source.empty:
                # 計算 Categorized Min/Max
                cat_stats = df_cat_source.groupby(['Station', 'Config', 'Category'])['CPK'].agg(['min', 'max']).reset_index()
                
                # 轉 Pivot Table (寬表)
                df_pivot = cat_stats.pivot(index=['Station', 'Config'], columns='Category', values=['min', 'max'])
                
                # 重新命名欄位
                new_columns = []
                for agg_type, cat_name in df_pivot.columns:
                    if "Category 1" in cat_name:
                        prefix = "Category1(X/Y/Rotation)"
                    elif "Category 2" in cat_name:
                        prefix = "Category2(height/ tilt)"
                    else:
                        prefix = cat_name
                    
                    suffix = "min Cpk" if agg_type == "min" else "max Cpk"
                    new_columns.append(f"{prefix} {suffix}")
                
                df_pivot.columns = new_columns
                df_pivot = df_pivot.reset_index()
                
                # 合併 Sample Size, Yield & Failed lists
                base_info = df_analysis[['Station', 'Config', 'Sample Size', 'Yield', 'Dims CPK < 1.33', 'Dims CPK < 1.0']]
                df_categorized_wide = pd.merge(base_info, df_pivot, on=['Station', 'Config'], how='left')
                
                # 定義輸出順序
                desired_cols = [
                    'Station', 'Config', 'Sample Size', 'Yield',
                    'Category1(X/Y/Rotation) min Cpk', 'Category1(X/Y/Rotation) max Cpk',
                    'Category2(height/ tilt) min Cpk', 'Category2(height/ tilt) max Cpk',
                    'Dims CPK < 1.33', 'Dims CPK < 1.0'
                ]
                final_cols = [c for c in desired_cols if c in df_categorized_wide.columns]
                df_categorized_wide = df_categorized_wide[final_cols]
                
                df_categorized_wide["Station"] = pd.Categorical(df_categorized_wide["Station"], TARGET_ORDER, ordered=True)
                df_categorized_wide = df_categorized_wide.sort_values(['Station', 'Config'])

            # Sorting others
            df_summary["Station"] = pd.Categorical(df_summary["Station"], TARGET_ORDER, ordered=True)
            df_summary = df_summary.sort_values(['Station', 'Config'])
            cols_sum = ['Station', 'Config', 'Sample Size', 'Yield', 'Min CPK Dim', 'Min CPK', 'Max CPK']
            df_summary = df_summary[[c for c in cols_sum if c in df_summary.columns]]

            df_analysis["Station"] = pd.Categorical(df_analysis["Station"], TARGET_ORDER, ordered=True)
            df_analysis = df_analysis.sort_values(['Station', 'Config'])
            cols_ana = ['Station', 'Config', 'Sample Size', 'Yield', 'Min CPK Dim', 'Min CPK', 'Max CPK', 'Dims CPK < 1.33', 'Dims CPK < 1.0']
            df_analysis = df_analysis[[c for c in cols_ana if c in df_analysis.columns]]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Yield", "CPK", "CPK by Config", "CPK Summary", "CPK Analysis", "Cpk analysis (categorized)"
    ])

    with tab1:
        st.dataframe(df_yield, use_container_width=True)
    with tab2:
        st.write(df_cpk.style.applymap(highlight_low_cpk, subset=["CPK"]))
    with tab3:
        st.write(df_cpk_config.style.applymap(highlight_low_cpk, subset=["CPK"]))
    with tab4:
        st.write("統計每個站點與 Config 下的極值")
        st.write(df_summary.style.applymap(highlight_low_cpk, subset=["Min CPK", "Max CPK"]))
    with tab5:
        st.write("詳細分析：列出 CPK 不達標的具體尺寸")
        st.write(df_analysis.style.applymap(highlight_low_cpk, subset=["Min CPK", "Max CPK"]))
    with tab6:
        st.write("分類分析 (Wide Format)")
        cat_cols = [c for c in df_categorized_wide.columns if "Cpk" in c and "Dims" not in c]
        st.write(df_categorized_wide.style.applymap(highlight_low_cpk, subset=cat_cols))

    # ===== Excel 匯出 =====
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_yield.to_excel(writer, sheet_name="Yield", index=False)
        df_cpk.to_excel(writer, sheet_name="CPK", index=False)
        df_cpk_config.to_excel(writer, sheet_name="CPK by Config", index=False)
        df_summary.to_excel(writer, sheet_name="CPK Summary", index=False)
        df_analysis.to_excel(writer, sheet_name="CPK Analysis", index=False)
        df_categorized_wide.to_excel(writer, sheet_name="Cpk analysis (categorized)", index=False)

        wb = writer.book
        # 新的顏色定義
        red = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006", "bold": True})    # < 0.67
        orange = wb.add_format({"bg_color": "#FFEB9C", "font_color": "#9C5700", "bold": True}) # < 1.0
        yellow = wb.add_format({"bg_color": "#FFFFCC", "font_color": "#808000", "bold": True}) # < 1.33
        
        def apply_cond_format(sheet_name, df, cols_to_check):
            if df.empty: return
            ws = writer.sheets[sheet_name]
            for col_name in cols_to_check:
                if col_name not in df.columns: continue
                col_letter = excel_col_letter(df.columns.get_loc(col_name))
                rng = f"{col_letter}2:{col_letter}{len(df)+1}"
                
                ws.conditional_format(rng, {"type": "cell", "criteria": "<", "value": 0.67, "format": red})
                ws.conditional_format(rng, {"type": "cell", "criteria": "between", "minimum": 0.67, "maximum": 0.9999, "format": orange})
                ws.conditional_format(rng, {"type": "cell", "criteria": "between", "minimum": 1.0, "maximum": 1.3299, "format": yellow})

        apply_cond_format("CPK", df_cpk, ["CPK"])
        apply_cond_format("CPK by Config", df_cpk_config, ["CPK"])
        apply_cond_format("CPK Summary", df_summary, ["Min CPK", "Max CPK"])
        apply_cond_format("CPK Analysis", df_analysis, ["Min CPK", "Max CPK"])
        
        cat_check_cols = [c for c in df_categorized_wide.columns if "Cpk" in c and "Dims" not in c]
        apply_cond_format("Cpk analysis (categorized)", df_categorized_wide, cat_check_cols)

    output.seek(0)
    st.download_button(
        "📥 下載 Excel",
        output,
        "IPQC_Report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )