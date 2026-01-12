import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================
# 1. 系統初始化
# =========================================================
st.set_page_config(page_title="台股 AI 深度全方位預測系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 核心運算引擎 (最高/最低雙向預估)
# =========================================================
def get_chips_weight(stock_id):
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start = (datetime.now() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        df_i = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_i.empty:
            net = df_i.tail(5)['buy'].sum() - df_i.tail(5)['sell'].sum()
            return (1.018, "✅ 法人籌碼偏多") if net > 0 else (0.982, "⚠️ 法人籌碼偏空")
    except:
        pass
    return 1.0, "ℹ️ 籌碼資料同步中"

def ai_full_engine(df, chip_f=1.0):
    """計算最高/最低價與獨立準確率"""
    if len(df) < 100: return [0]*8
    
    df_c = df.copy()
    if isinstance(df_c.columns, pd.MultiIndex): df_c.columns = df_c.columns.get_level_values(0)
    
    # 漲跌幅慣性
    df_c['h_pct'] = (df_c['High'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c['l_pct'] = (df_c['Low'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    vol = df_c['Close'].pct_change().tail(20).std()
    
    # 分位數定義
    q_h1, q_l1 = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    q_h5, q_l5 = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    
    # 預估值
    h1 = df_c['h_pct'].tail(80).quantile(q_h1) * chip_f
    l1 = df_c['l_pct'].tail(80).quantile(q_l1) / chip_f
    h5 = df_c['h_pct'].tail(80).quantile(q_h5) * chip_f
    l5 = df_c['l_pct'].tail(80).quantile(q_l5) / chip_f
    
    # 回測命中率 (過去 20 天)
    test_days = 20
    hits = {"h1":0, "l1":0, "h5":0, "l5":0}
    
    for i in range(test_days):
        idx = -(test_days + 5) + i
        train = df_c.iloc[:idx]
        if len(train) < 60: continue
        
        pc = train['Close'].iloc[-1]
        # 模擬當時預估
        ph1 = train['h_pct'].tail(60).quantile(q_h1) * chip_f
        pl1 = train['l_pct'].tail(60).quantile(q_l1) / chip_f
        ph5 = train['h_pct'].tail(60).quantile(q_h5) * chip_f
        pl5 = train['l_pct'].tail(60).quantile(q_l5) / chip_f
        
        if df_c['High'].iloc[idx] >= pc*(1+ph1): hits["h1"] += 1
        if df_c['Low'].iloc[idx] <= pc*(1+pl1): hits["l1"] += 1
        if df_c['High'].iloc[idx:idx+5].max() >= pc*(1+ph5): hits["h5"] += 1
        if df_c['Low'].iloc[idx:idx+5].min() <= pc*(1+pl5): hits["l5"] += 1
            
    return h1, l1, h5, l5, (hits["h1"]/test_days)*100, (hits["l1"]/test_days)*100, (hits["h5"]/test_days)*100, (hits["l5"]/test_days)*100

# =========================================================
# 3. 頁面邏輯
# =========================================================

if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 深度全方位預測系統")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### ⚡ 盤中即時預測")
        if st.button("進入盤中監控"): navigate_to("realtime")
    with c2:
        st.success("### 📊 深度預估分析")
        if st.button("進入深度頁面"): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時高低點監控")
    sid = st.text_input("輸入代碼:")
    if sid:
        df_rt = yf.download(f"{sid}.TW", period="1d", interval="1m", progress=False)
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            now_p = df_rt['Close'].iloc[-1]
            st.metric("盤中現價", f"{now_p:.2f}")
            df_h = yf.download(f"{sid}.TW", period="120d", progress=False)
            h1, l1, _, _, acc_h1, acc_l1, _, _ = ai_full_engine(df_h)
            
            c1, c2 = st.columns(2)
            c1.error(f"🎯 今日預估最高: {now_p*(1+h1):.2f} (準確率: {acc_h1:.1f}%)")
            c2.success(f"🎯 今日預估最低: {now_p*(1+l1):.2f} (準確率: {acc_l1:.1f}%)")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日與五日高低點深度預判")
    sid = st.text_input("輸入代碼 (例: 2330):")
    if sid:
        df = yf.download(f"{sid}.TW", period="250d", progress=False)
        if df.empty: df = yf.download(f"{sid}.TWO", period="250d", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            curr_c = float(df['Close'].iloc[-1])
            chip_f, chip_m = get_chips_weight(sid)
            h1, l1, h5, l5, ah1, al1, ah5, al5 = ai_full_engine(df, chip_f)

            st.subheader(f"🏠 標的：{sid} | 最新收盤：{curr_c:.2f}")
            st.write(f"🧬 {chip_m}")

            # --- 第一排：隔日最高/最低 ---
            st.markdown("### 📅 隔日(T+1) 預估範圍")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background:#fff5f5; padding:15px; border-radius:10px; border:1px solid #ffcccc'><h4>📈 隔日預估最高</h4><h2 style='color:#e63946;'>{curr_c*(1+h1):.2f}</h2><p>準確率: {ah1:.1f}%</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background:#f6fff6; padding:15px; border-radius:10px; border:1px solid #ccffcc'><h4>📉 隔日預估最低</h4><h2 style='color:#2a9d8f;'>{curr_c*(1+l1):.2f}</h2><p>準確率: {al1:.1f}%</p></div>", unsafe_allow_html=True)

            # --- 第二排：五日最高/最低 ---
            st.markdown("### 🚩 五日(T+5) 預估範圍")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"<div style='background:#f0f7ff; padding:15px; border-radius:10px; border:1px solid #cce3ff'><h4>🚀 五日預估最高</h4><h2 style='color:#0077b6;'>{curr_c*(1+h5):.2f}</h2><p>準確率: {ah5:.1f}%</p></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div style='background:#fffdf0; padding:15px; border-radius:10px; border:1px solid #ffecb3'><h4>⚓ 五日預估最低</h4><h2 style='color:#d4a017;'>{curr_c*(1+l5):.2f}</h2><p>準確率: {al5:.1f}%</p></div>", unsafe_allow_html=True)

            # --- 價量圖 ---
            st.divider()
            st.write("### 📈 Price & Color-Volume Analysis")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            p_df = df.tail(40).copy()
            ax1.plot(p_df.index, p_df['Close'], color="#1f77b4", lw=2)
            ax1.axhline(curr_c*(1+h1), color='red', ls='--', alpha=0.7, label="T+1 High")
            ax1.axhline(curr_c*(1+l1), color='green', ls='--', alpha=0.7, label="T+1 Low")
            ax1.legend()
            
            p_df['diff'] = p_df['Close'].diff()
            v_colors = ['#e63946' if x >= 0 else '#2a9d8f' for x in p_df['diff']]
            ax2.bar(p_df.index, p_df['Volume'], color=v_colors, alpha=0.8)
            st.pyplot(fig)
            st.markdown("**說明：** 紅色虛線為預估最高，綠色虛線為預估最低。成交量紅色代表上漲，綠色代表下跌。")
