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
# 排除清單 (Exclusion List for Analysis)
# =========================
# 這些尺寸將不會出現在 Analysis 與 Summary 的統計結果中
EXCLUDED_DIMS_ANALYSIS = [
    "CPK301", "CPK303", 
    "CPK301-X", "CPK301-Y", 
    "CPK205", "CPK203", 
    "CPK204-Tilt",
    "CPK301-1"  # [新增] 排除此尺寸
]

# =========================
# 站點順序 (Station Order)
# =========================
TARGET_ORDER = [
    "MLA assy installation",
    "Mirror attachment",
    "Barrel attachment",
    "Barrel attachment (SHC)",    
    "Condenser lens attach",
    "LED Module  attachment",
    "LED Module attachment (SHC)", 
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
    "LCOS GRAPHITE ATTACH",
    "PANEL FLEX ASSY",
    "PANEL FLEX ASSY-2",
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
    "Barrel attachment (SHC)": {
        "Category 1": ["CPK206", "CPK207", "CPK208"],
        "Category 2": ["CPK201", "CPK203", "CPK204", "CPK205"]
    },
    "Condenser lens attach": {
        "Category 1": ["CPK403"]
    },
    "LED Module  attachment": {
        "Category 1": ["CPK403"]
    },
    "LED Module attachment (SHC)": {
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
    "PANEL FLEX ASSY": {
        "Category 1": ["ALL"]
    },
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
    
    # 根據是否為 NaN 來決定回傳值 (單邊公差邏輯)
    if not pd.isna(cpu) and not pd.isna(cpl):
        return min(cpu, cpl)
    elif not pd.isna(cpu):
        return cpu # Cpku
    elif not pd.isna(cpl):
        return cpl # Cpkl
    else:
        return np.nan

def excel_col_letter(idx):
    letters = ""
    while idx >= 0:
        idx, r = divmod(idx, 26)
        letters = chr(65 + r) + letters
        idx -= 1
    return letters

def highlight_low_cpk(val):
    try:
        v = float(val)
        if v < 0.67:
            return "background-color:#ffc7ce;color:#9c0006;font-weight:bold"
        elif v < 1.0:
            return "background-color:#ffeb9c;color:#9c5700;font-weight:bold"
        elif v < 1.33:
            return "background-color:#ffffcc;color:#808000;font-weight:bold"
    except:
        pass
    return ""

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

def get_failed_dims_detailed(g):
    cat1_133, cat1_100 = [], []
    cat2_133, cat2_100 = [], []
    all_133, all_100 = [], []
    
    for _, row in g.iterrows():
        cat = identify_category(row)
        dim = row['Dim No']
        val = row['CPK']
        
        if val < 1.33:
            all_133.append(dim)
            if cat == "Category 1":
                cat1_133.append(dim)
            elif cat == "Category 2":
                cat2_133.append(dim)
        
        if val < 1.0:
            all_100.append(dim)
            if cat == "Category 1":
                cat1_100.append(dim)
            elif cat == "Category 2":
                cat2_100.append(dim)
                
    return pd.Series({
        'Dims CPK < 1.33': ", ".join(map(str, pd.unique(all_133))),
        'Dims CPK < 1.0': ", ".join(map(str, pd.unique(all_100))),
        'Category1(X/Y/Rotation) Dims CPK < 1.33': ", ".join(map(str, pd.unique(cat1_133))),
        'Category1(X/Y/Rotation) Dims CPK < 1.0': ", ".join(map(str, pd.unique(cat1_100))),
        'Category2(height/ tilt) Dims CPK < 1.33': ", ".join(map(str, pd.unique(cat2_133))),
        'Category2(height/ tilt) Dims CPK < 1.0': ", ".join(map(str, pd.unique(cat2_100)))
    })

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

def process_yield_combined(station, df, target_configs, label):
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
    
    subset = subset[subset["Config"].isin(target_configs)]
    subset = subset[subset["Result"].isin(["OK", "NG"])]
    
    if subset.empty:
        return []
    
    ok = (subset["Result"] == "OK").sum()
    ng = (subset["Result"] == "NG").sum()
    
    return [{
        "Station": station,
        "Config": label,
        "OK": ok,
        "NG": ng
    }]

# =========================
# CPK Extraction Logic
# =========================
def _extract_data_for_cpk(df):
    def find_row_index(keywords):
        for i in range(min(60, len(df))):
            row_vals = [str(v).strip().lower() for v in df.iloc[i, :10]]
            if any(k in row_vals for k in keywords):
                return i
        return -1

    dim_row = find_row_index(["dim. no", "dim no", "dim.no", "dim_no", "dim. no."])
    usl_row = find_row_index(["usl", "u.s.l", "u.s.l."])
    lsl_row = find_row_index(["lsl", "l.s.l", "l.s.l."])
    dist_row = find_row_index(["distribution type", "distributiontype"])
    config_header_row = find_row_index(["config"])

    if dim_row == -1:
        return None, None, None, None, None

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

    dist_types = {}
    if dist_row != -1:
        for i, v in enumerate(df.iloc[dist_row]):
            if i in dim_cols:
                dist_types[i] = str(v).strip()

    start_row_candidates = [dim_row, usl_row, lsl_row]
    if config_header_row != -1:
        start_row_candidates.append(config_header_row)
    if dist_row != -1:
        start_row_candidates.append(dist_row)
    
    start = max(start_row_candidates) + 1
    data = df.iloc[start:].copy()

    date_col = -1
    for c in range(min(15, data.shape[1])):
        if data.iloc[:, c].astype(str).str.contains(r"202\d-\d{2}-\d{2}").any():
            date_col = c
            break
    if date_col == -1:
        return None, None, None, None, None

    data["Date"] = data.iloc[:, date_col].astype(str).str.extract(r"(202\d-\d{2}-\d{2})")[0]
    data = data.dropna(subset=["Date"])

    config_col_idx = 1
    if data.shape[1] > config_col_idx:
        data["Config"] = data.iloc[:, config_col_idx].astype(str).fillna("")
    else:
        data["Config"] = ""

    return data, dim_cols, usls, lsls, dist_types

# =========================
# CPK Processing
# =========================
def process_cpk(station, df):
    data, dim_cols, usls, lsls, dist_types = _extract_data_for_cpk(df)
    if data is None: 
        return []

    results = []
    for (date, config_val), g in data.groupby(["Date", "Config"]):
        for idx, dim in dim_cols.items():
            vals = g.iloc[:, idx]
            
            usl = usls.get(idx)
            lsl = lsls.get(idx)
            dist_type = dist_types.get(idx, "").lower() if dist_types else ""
            
            # 單邊公差判定
            is_single_usl = "singleside-usl" in dist_type
            is_single_lsl = "singleside-lsl" in dist_type
            
            calc_lsl = np.nan if is_single_usl else lsl
            calc_usl = np.nan if is_single_lsl else usl
            
            cpk = calculate_cpk(vals, calc_usl, calc_lsl)
            n = pd.to_numeric(vals, errors="coerce").dropna().size
            if n > 0:
                results.append({
                    "Station": station,
                    "Dim No": dim,
                    "Config": config_val,
                    "Date": date,
                    "Sample Size": n,
                    "USL": usl,
                    "LSL": lsl,
                    "CPK": round(cpk, 3) if not pd.isna(cpk) else ""
                })
    return results

def process_cpk_by_config(station, df):
    data, dim_cols, usls, lsls, dist_types = _extract_data_for_cpk(df)
    if data is None: 
        return []

    results = []
    for config_val, g in data.groupby("Config"):
        for idx, dim in dim_cols.items():
            vals = g.iloc[:, idx]
            
            usl = usls.get(idx)
            lsl = lsls.get(idx)
            dist_type = dist_types.get(idx, "").lower() if dist_types else ""
            
            is_single_usl = "singleside-usl" in dist_type
            is_single_lsl = "singleside-lsl" in dist_type
            
            calc_lsl = np.nan if is_single_usl else lsl
            calc_usl = np.nan if is_single_lsl else usl
            
            cpk = calculate_cpk(vals, calc_usl, calc_lsl)
            n = pd.to_numeric(vals, errors="coerce").dropna().size
            if n > 0:
                results.append({
                    "Station": station,
                    "Dim No": dim,
                    "Config": config_val,
                    "Sample Size": n,
                    "USL": usl, # 報表顯示原始值
                    "LSL": lsl, # 報表顯示原始值
                    "CPK": round(cpk, 3) if not pd.isna(cpk) else ""
                })
    return results

def process_cpk_combined(station, df, target_configs, label):
    data, dim_cols, usls, lsls, dist_types = _extract_data_for_cpk(df)
    if data is None: 
        return []

    data = data[data["Config"].isin(target_configs)].copy()
    if data.empty:
        return []
    
    data["Config"] = label

    results = []
    for config_val, g in data.groupby("Config"):
        for idx, dim in dim_cols.items():
            vals = g.iloc[:, idx]
            
            usl = usls.get(idx)
            lsl = lsls.get(idx)
            dist_type = dist_types.get(idx, "").lower() if dist_types else ""
            
            is_single_usl = "singleside-usl" in dist_type
            is_single_lsl = "singleside-lsl" in dist_type
            
            calc_lsl = np.nan if is_single_usl else lsl
            calc_usl = np.nan if is_single_lsl else usl
            
            cpk = calculate_cpk(vals, calc_usl, calc_lsl)
            n = pd.to_numeric(vals, errors="coerce").dropna().size
            if n > 0:
                results.append({
                    "Station": station,
                    "Dim No": dim,
                    "Config": config_val,
                    "Sample Size": n,
                    "USL": usl,
                    "LSL": lsl,
                    "CPK": round(cpk, 3) if not pd.isna(cpk) else ""
                })
    return results

# =========================
# Streamlit 主流程
# =========================
uploaded = st.file_uploader("📂 上傳 Excel (.xlsx)", type=["xlsx"])

if uploaded:
    xls = pd.ExcelFile(uploaded)
    yield_list, cpk_list, cpk_config_list, cpk_combined_list = [], [], [], []
    yield_config_counts = []
    yield_combined_counts = []

    sorted_target_keys = sorted(TARGET_MAP.keys(), key=len, reverse=True)

    for sheet in xls.sheet_names:
        norm = normalize_name(sheet)
        if "slice" in norm or "slic" in norm:
            continue

        station = next((TARGET_MAP[k] for k in sorted_target_keys if k in norm), None)
        if not station:
            continue
        df = pd.read_excel(uploaded, sheet_name=sheet, header=None)
        
        y = process_yield(station, df)
        if y:
            yield_list.append(y)
            
        yield_config_counts.extend(process_yield_by_config(station, df))
        yield_combined_counts.extend(
            process_yield_combined(station, df, ["SHD", "SHA", "SHC"], "SHD+SHA+SHC")
        )
        
        cpk_list.extend(process_cpk(station, df))
        cpk_config_list.extend(process_cpk_by_config(station, df))
        cpk_combined_list.extend(
            process_cpk_combined(station, df, ["SHD", "SHA", "SHC"], "SHD+SHA+SHC")
        )

    # DataFrames
    df_yield = pd.DataFrame(yield_list)
    df_cpk = pd.DataFrame(cpk_list)
    df_cpk_config = pd.DataFrame(cpk_config_list)
    df_cpk_combined = pd.DataFrame(cpk_combined_list)
    
    # Aggregated Yields
    df_yield_agg = pd.DataFrame()
    if yield_config_counts:
        df_counts = pd.DataFrame(yield_config_counts)
        df_yield_agg = df_counts.groupby(["Station", "Config"], as_index=False)[["OK", "NG"]].sum()
        df_yield_agg["Total"] = df_yield_agg["OK"] + df_yield_agg["NG"]
        df_yield_agg["Yield Raw"] = df_yield_agg.apply(
            lambda r: r["OK"] / r["Total"] if r["Total"] > 0 else 0, axis=1
        )
        
    df_yield_combined_agg = pd.DataFrame()
    if yield_combined_counts:
        df_counts_comb = pd.DataFrame(yield_combined_counts)
        df_yield_combined_agg = df_counts_comb.groupby(["Station", "Config"], as_index=False)[["OK", "NG"]].sum()
        df_yield_combined_agg["Total"] = df_yield_combined_agg["OK"] + df_yield_combined_agg["NG"]
        df_yield_combined_agg["Yield Raw"] = df_yield_combined_agg.apply(
            lambda r: r["OK"] / r["Total"] if r["Total"] > 0 else 0, axis=1
        )

    # --- Generate Wide Format Tables ---
    def generate_categorized_table(df_cpk_data, df_yield_data):
        if df_cpk_data.empty:
            return pd.DataFrame()
            
        df_for_stats = df_cpk_data.copy()
        
        # 1. 移除 FAI 尺寸
        df_for_stats = df_for_stats[~df_for_stats['Dim No'].astype(str).str.strip().str.lower().str.startswith('fai')]
        
        # 2. 移除排除清單中的尺寸
        exclude_lower = [x.lower() for x in EXCLUDED_DIMS_ANALYSIS]
        df_for_stats = df_for_stats[~df_for_stats['Dim No'].astype(str).str.strip().str.lower().isin(exclude_lower)]
        
        df_for_stats['CPK'] = pd.to_numeric(df_for_stats['CPK'], errors='coerce')
        df_for_stats = df_for_stats.dropna(subset=['CPK'])
        
        if df_for_stats.empty:
            return pd.DataFrame()

        failed_lists = df_for_stats.groupby(['Station', 'Config']).apply(get_failed_dims_detailed).reset_index()
        
        df_for_stats['Category'] = df_for_stats.apply(identify_category, axis=1)
        df_for_stats = df_for_stats.dropna(subset=['Category'])
        
        if df_for_stats.empty:
            return pd.DataFrame()

        cat_stats = df_for_stats.groupby(['Station', 'Config', 'Category'])['CPK'].agg(['min', 'max']).reset_index()
        df_pivot = cat_stats.pivot(index=['Station', 'Config'], columns='Category', values=['min', 'max'])
        
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
        
        result = pd.merge(df_pivot, failed_lists, on=['Station', 'Config'], how='left')
        
        if not df_yield_data.empty:
            yield_info = df_yield_data[["Station", "Config", "Yield Raw", "Total"]]
            result = pd.merge(result, yield_info, on=["Station", "Config"], how="left")
            result["Yield"] = result["Yield Raw"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
            result["Sample Size"] = result["Total"].fillna(0).astype(int)
        else:
            result["Yield"] = ""
            result["Sample Size"] = 0
            
        cols = [
            'Station', 'Config', 'Sample Size', 'Yield',
            'Category1(X/Y/Rotation) min Cpk', 'Category1(X/Y/Rotation) max Cpk',
            'Category2(height/ tilt) min Cpk', 'Category2(height/ tilt) max Cpk',
            'Dims CPK < 1.33', 'Dims CPK < 1.0',
            'Category1(X/Y/Rotation) Dims CPK < 1.33', 'Category1(X/Y/Rotation) Dims CPK < 1.0',
            'Category2(height/ tilt) Dims CPK < 1.33', 'Category2(height/ tilt) Dims CPK < 1.0'
        ]
        final_cols = [c for c in cols if c in result.columns]
        result = result[final_cols]
        
        result["Station"] = pd.Categorical(result["Station"], TARGET_ORDER, ordered=True)
        result = result.sort_values(['Station', 'Config'])
        return result

    # 1. Summary (Original Long)
    df_summary = pd.DataFrame()
    df_analysis = pd.DataFrame()
    
    if not df_cpk_config.empty:
        df_s = df_cpk_config.copy()
        
        # FAI Filter
        df_s = df_s[~df_s['Dim No'].astype(str).str.strip().str.lower().str.startswith('fai')]
        # Explicit Exclusion Filter
        ex_lower = [x.lower() for x in EXCLUDED_DIMS_ANALYSIS]
        df_s = df_s[~df_s['Dim No'].astype(str).str.strip().str.lower().isin(ex_lower)]
        
        df_s['CPK'] = pd.to_numeric(df_s['CPK'], errors='coerce')
        df_s = df_s.dropna(subset=['CPK'])
        
        if not df_s.empty:
            stats = df_s.groupby(['Station', 'Config'])['CPK'].agg(['min', 'max']).reset_index()
            loc_min = df_s.groupby(['Station', 'Config'])['CPK'].idxmin()
            min_dims = df_s.loc[loc_min, ['Station', 'Config', 'Dim No']].rename(columns={'Dim No': 'Min CPK Dim'})
            
            df_summary = pd.merge(stats, min_dims, on=['Station', 'Config'], how='left')
            df_summary.columns = ['Station', 'Config', 'Min CPK', 'Max CPK', 'Min CPK Dim']
            
            fails = df_s.groupby(['Station', 'Config']).apply(get_failed_dims_detailed).reset_index()
            
            if not df_yield_agg.empty:
                yinfo = df_yield_agg[["Station", "Config", "Yield Raw", "Total"]]
                df_summary = pd.merge(df_summary, yinfo, on=["Station", "Config"], how="left")
            else:
                df_summary["Yield Raw"] = np.nan
                df_summary["Total"] = 0
            
            df_summary["Yield"] = df_summary["Yield Raw"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
            df_summary["Sample Size"] = df_summary["Total"].fillna(0).astype(int)
            
            df_analysis = pd.merge(df_summary, fails, on=['Station', 'Config'], how='left')
            
            cols_s = ['Station', 'Config', 'Sample Size', 'Yield', 'Min CPK Dim', 'Min CPK', 'Max CPK']
            df_summary = df_summary[[c for c in cols_s if c in df_summary.columns]]
            
            cols_a = [
                'Station', 'Config', 'Sample Size', 'Yield', 'Min CPK Dim', 'Min CPK', 'Max CPK', 
                'Dims CPK < 1.33', 'Dims CPK < 1.0',
                'Category1(X/Y/Rotation) Dims CPK < 1.33', 'Category1(X/Y/Rotation) Dims CPK < 1.0',
                'Category2(height/ tilt) Dims CPK < 1.33', 'Category2(height/ tilt) Dims CPK < 1.0'
            ]
            final_cols_a = [c for c in cols_a if c in df_analysis.columns]
            df_analysis = df_analysis[final_cols_a]
            
            df_summary["Station"] = pd.Categorical(df_summary["Station"], TARGET_ORDER, ordered=True)
            df_summary = df_summary.sort_values(['Station', 'Config'])
            df_analysis["Station"] = pd.Categorical(df_analysis["Station"], TARGET_ORDER, ordered=True)
            df_analysis = df_analysis.sort_values(['Station', 'Config'])

    # 2. Categorized Wide Tables
    df_categorized_wide = generate_categorized_table(df_cpk_config, df_yield_agg)
    df_combined_wide = generate_categorized_table(df_cpk_combined, df_yield_combined_agg)

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

    tabs = st.tabs([
        "Yield", "CPK", "CPK by Config", "CPK Summary", "CPK Analysis", 
        "Cpk analysis (categorized)", "CPK Analysis (SHD+SHA+SHC)"
    ])

    with tabs[0]: 
        if not df_yield.empty:
            st.dataframe(df_yield, use_container_width=True)
        else:
            st.write(df_yield)
    with tabs[1]: 
        if not df_cpk.empty:
            st.write(df_cpk.style.applymap(highlight_low_cpk, subset=["CPK"]))
        else:
            st.write(df_cpk)
    with tabs[2]: 
        if not df_cpk_config.empty:
            st.write(df_cpk_config.style.applymap(highlight_low_cpk, subset=["CPK"]))
        else:
            st.write(df_cpk_config)
    with tabs[3]: 
        if not df_summary.empty:
            st.write(df_summary.style.applymap(highlight_low_cpk, subset=["Min CPK", "Max CPK"]))
        else:
            st.write(df_summary)
    with tabs[4]: 
        if not df_analysis.empty:
            st.write(df_analysis.style.applymap(highlight_low_cpk, subset=["Min CPK", "Max CPK"]))
        else:
            st.write(df_analysis)
    with tabs[5]: 
        if not df_categorized_wide.empty:
            st.write(df_categorized_wide.style.applymap(highlight_low_cpk, subset=[c for c in df_categorized_wide.columns if "Cpk" in c and "Dims" not in c]))
        else:
            st.write(df_categorized_wide)
    with tabs[6]: 
        if not df_combined_wide.empty:
            st.write(df_combined_wide.style.applymap(highlight_low_cpk, subset=[c for c in df_combined_wide.columns if "Cpk" in c and "Dims" not in c]))
        else:
            st.write(df_combined_wide)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_yield.to_excel(writer, sheet_name="Yield", index=False)
        df_cpk.to_excel(writer, sheet_name="CPK", index=False)
        df_cpk_config.to_excel(writer, sheet_name="CPK by Config", index=False)
        df_summary.to_excel(writer, sheet_name="CPK Summary", index=False)
        df_analysis.to_excel(writer, sheet_name="CPK Analysis", index=False)
        df_categorized_wide.to_excel(writer, sheet_name="Cpk analysis (categorized)", index=False)
        df_combined_wide.to_excel(writer, sheet_name="CPK Analysis (Combined)", index=False)

        wb = writer.book
        red = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006", "bold": True})
        orange = wb.add_format({"bg_color": "#FFEB9C", "font_color": "#9C5700", "bold": True})
        yellow = wb.add_format({"bg_color": "#FFFFCC", "font_color": "#808000", "bold": True})
        
        def apply_format_and_freeze(sheet_name, df, cols_to_check):
            if df.empty: return
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
            
            for i, col_name in enumerate(df.columns):
                if "Dims CPK <" in col_name:
                    ws.set_column(i, i, 50)
            
            for col_name in cols_to_check:
                if col_name not in df.columns: continue
                col_letter = excel_col_letter(df.columns.get_loc(col_name))
                rng = f"{col_letter}2:{col_letter}{len(df)+1}"
                ws.conditional_format(rng, {"type": "cell", "criteria": "<", "value": 0.67, "format": red})
                ws.conditional_format(rng, {"type": "cell", "criteria": "between", "minimum": 0.67, "maximum": 0.9999, "format": orange})
                ws.conditional_format(rng, {"type": "cell", "criteria": "between", "minimum": 1.0, "maximum": 1.3299, "format": yellow})

        apply_format_and_freeze("CPK", df_cpk, ["CPK"])
        apply_format_and_freeze("CPK by Config", df_cpk_config, ["CPK"])
        apply_format_and_freeze("CPK Summary", df_summary, ["Min CPK", "Max CPK"])
        apply_format_and_freeze("CPK Analysis", df_analysis, ["Min CPK", "Max CPK"])
        
        cat_check_cols = [c for c in df_categorized_wide.columns if "Cpk" in c and "Dims" not in c]
        apply_format_and_freeze("Cpk analysis (categorized)", df_categorized_wide, cat_check_cols)
        
        comb_check_cols = [c for c in df_combined_wide.columns if "Cpk" in c and "Dims" not in c]
        apply_format_and_freeze("CPK Analysis (Combined)", df_combined_wide, comb_check_cols)
        
        apply_format_and_freeze("Yield", df_yield, [])

    output.seek(0)
    st.download_button(
        "📥 下載 Excel",
        output,
        "IPQC_Report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )