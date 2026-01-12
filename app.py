import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================
# 1. 系統初始化與導航 (還原原始結構)
# =========================================================
st.set_page_config(page_title="台股 AI 預測系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 核心運算引擎 (還原 FinMind 整合與 Volatility 慣性)
# =========================================================
def get_institutional_chips(stock_id):
    """計算籌碼修正因子 (FinMind)"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start_dt = (datetime.now() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        chip_weight = 1.0 
        msg = "籌碼狀態：偏向中性"
        if not inst_df.empty:
            net = inst_df.tail(9)['buy'].sum() - inst_df.tail(9)['sell'].sum()
            if net > 0:
                chip_weight += 0.018
                msg = "✅ 籌碼強勢：法人近期買超"
            else:
                chip_weight -= 0.018
                msg = "⚠️ 籌碼轉弱：法人近期調節"
        return round(chip_weight, 4), msg
    except:
        return 1.0, "⚠️ 籌碼資料同步中..."

def ai_dynamic_forecast(df, chip_f=1.0):
    """AI 動態分位數預測：考慮波動慣性與法人籌碼"""
    if len(df) < 100: return [0]*8
    
    # 計算波動慣性 (Volatility)
    vol = df['Close'].pct_change().tail(20).std()
    
    # 動態分位數邏輯
    h1_q, l1_q = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    h5_q, l5_q = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    
    df_c = df.copy()
    if isinstance(df_c.columns, pd.MultiIndex): df_c.columns = df_c.columns.get_level_values(0)
    
    # 計算漲跌幅百分比
    df_c['h_pct'] = (df_c['High'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c['l_pct'] = (df_c['Low'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c = df_c.dropna(subset=['h_pct', 'l_pct'])

    # 產出預估幅
    h1 = df_c['h_pct'].tail(80).quantile(h1_q) * chip_f
    l1 = df_c['l_pct'].tail(80).quantile(l1_q) / chip_f
    h5 = df_c['h_pct'].tail(80).quantile(h5_q) * chip_f
    l5 = df_c['l_pct'].tail(80).quantile(l5_q) / chip_f
    
    # --- 回測準確率 (過去 20 天) ---
    test_days = 20
    hits = {"h1":0, "l1":0, "h5":0, "l5":0}
    for i in range(test_days):
        idx = -(test_days) + i
        if idx >= 0: continue
        train = df_c.iloc[:idx]
        pc = train['Close'].iloc[-1]
        if df_c['High'].iloc[idx] >= pc * (1 + h1): hits["h1"] += 1
        if df_c['Low'].iloc[idx] <= pc * (1 + l1): hits["l1"] += 1
        if df_c['High'].iloc[idx:idx+5].max() >= pc * (1 + h5): hits["h5"] += 1
        if df_c['Low'].iloc[idx:idx+5].min() <= pc * (1 + l5): hits["l5"] += 1
            
    return h1, l1, h5, l5, (hits["h1"]/test_days)*100, (hits["l1"]/test_days)*100, (hits["h5"]/test_days)*100, (hits["l5"]/test_days)*100

# =========================================================
# 3. 頁面渲染與排版 (還原您的原始設計)
# =========================================================

# --- A. 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子動態回測系統")
    st.info("請選擇功能模式：")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚡ 盤中即時預測")
        if st.button("進入盤中監控", use_container_width=True): navigate_to("realtime")
    with col2:
        st.markdown("### 📊 隔日深度回測")
        if st.button("進入深度預判", use_container_width=True): navigate_to("forecast")

# --- B. 隔日深度回測頁面 ---
elif st.session_state.mode == "forecast":
    st.title("📊 隔日與五日深度預判分析")
    if st.button("🏠 回到首頁"): navigate_to("home")
    
    sid = st.text_input("輸入股票代碼 (例: 2330):")
    if sid:
        with st.spinner("正在進行 AI 運算..."):
            df = yf.download(f"{sid}.TW", period="250d", progress=False)
            if df.empty: df = yf.download(f"{sid}.TWO", period="250d", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                curr_c = float(df['Close'].iloc[-1])
                chip_f, chip_msg = get_institutional_chips(sid)
                h1, l1, h5, l5, ah1, al1, ah5, al5 = ai_dynamic_forecast(df, chip_f)

                # --- 頂部摘要 ---
                st.subheader(f"🏠 標的：{sid} | 最新收盤：{curr_c:.2f}")
                st.write(f"🧬 {chip_msg}")

                st.divider()
                # --- 數據盒子：隔日與五日預估價格 (還原顏色) ---
                st.markdown("### 🎯 預估目標價格與命中率")
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"📅 隔日預估最高: {curr_c*(1+h1):.2f} | 準確率: {ah1:.1f}%")
                    st.info(f"🚩 五日預估最高: {curr_c*(1+h5):.2f} | 準確率: {ah5:.1f}%")
                with c2:
                    st.success(f"📅 隔日預估最低: {curr_c*(1+l1):.2f} | 準確率: {al1:.1f}%")
                    st.success(f"⚓ 五日預估最低: {curr_c*(1+l5):.2f} | 準確率: {al5:.1f}%")

                # --- 實戰操作建議 ---
                st.divider()
                st.markdown("### ⚡ 當沖策略建議點位")
                s1, s2, s3 = st.columns(3)
                s1.warning(f"💡 建議買入位\n\n**{curr_c*(1+l1*0.5):.2f}**")
                s2.error(f"🚀 建議停利位\n\n**{curr_c*(1+h1*0.96):.2f}**")
                s3.info(f"🛑 建議停損位\n\n**{curr_c*0.985:.2f}**")

                # --- 彩色價量圖表 (修正跑不出來的問題) ---
                st.divider()
                st.write("### 📈 Price & Color-Volume Analysis")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                p_df = df.tail(40).copy()
                ax1.plot(p_df.index, p_df['Close'], color="#1f77b4", lw=2, label="Price")
                ax1.axhline(curr_c*(1+h1), color='red', ls='--', label="T+1 High")
                ax1.axhline(curr_c*(1+l1), color='green', ls='--', label="T+1 Low")
                ax1.legend()

                # 彩色成交量邏輯：今日收盤 >= 昨日收盤 為紅，否則為綠
                colors = ['#e63946' if p_df['Close'].iloc[i] >= p_df['Close'].iloc[i-1] else '#2a9d8f' for i in range(len(p_df))]
                ax2.bar(p_df.index, p_df['Volume'], color=colors, alpha=0.8)
                
                st.pyplot(fig)
                st.markdown("**📌 圖表註解：** 紅色虛線為預估最高點，綠色虛線為預估最低點。成交量柱狀：紅色代表上漲，綠色代表下跌。")
            else:
                st.error("查無數據，請確認代碼是否輸入正確。")
