import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# --- 系統配置 ---
st.set_page_config(page_title="台股 AI 深度全方位預測系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 核心運算引擎 ---
def ai_full_engine(df, chip_f=1.0):
    """計算最高/最低價與獨立準確率，並修正 KeyError 問題"""
    # 確保數據足夠
    if len(df) < 120: return [0]*8
    
    # 建立必要的欄位，避免 KeyError
    df_c = df.copy()
    if isinstance(df_c.columns, pd.MultiIndex): 
        df_c.columns = df_c.columns.get_level_values(0)
    
    # 計算漲跌幅慣性 (基於前日收盤)
    df_c['h_pct'] = (df_c['High'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c['l_pct'] = (df_c['Low'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c = df_c.dropna(subset=['h_pct', 'l_pct']) # 移除空值
    
    vol = df_c['Close'].pct_change().tail(20).std()
    
    # 動態分位數邏輯
    q_h1, q_l1 = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    q_h5, q_l5 = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    
    # 1. 當前預估值
    h1 = df_c['h_pct'].tail(100).quantile(q_h1) * chip_f
    l1 = df_c['l_pct'].tail(100).quantile(q_l1) / chip_f
    h5 = df_c['h_pct'].tail(100).quantile(q_h5) * chip_f
    l5 = df_c['l_pct'].tail(100).quantile(q_l5) / chip_f
    
    # 2. 歷史命中率回測 (過去 20 天)
    test_days = 20
    hits = {"h1":0, "l1":0, "h5":0, "l5":0}
    
    for i in range(test_days):
        # 確保回測視窗正確
        idx = -(test_days) + i 
        if idx >= 0: continue # 安全檢查
        
        # 模擬當時的基準與數據集
        train_data = df_c.iloc[:idx]
        if len(train_data) < 60: continue
        
        prev_close = train_data['Close'].iloc[-1]
        
        # 檢查隔日是否觸及
        actual_h = df_c['High'].iloc[idx]
        actual_l = df_c['Low'].iloc[idx]
        
        # 模擬當時的預估點
        p_h1 = train_data['h_pct'].tail(60).quantile(q_h1) * chip_f
        p_l1 = train_data['l_pct'].tail(60).quantile(q_l1) / chip_f
        
        if actual_h >= prev_close * (1 + p_h1): hits["h1"] += 1
        if actual_l <= prev_close * (1 + p_l1): hits["l1"] += 1
        
        # 五日回測 (僅在有足夠未來數據時執行)
        if idx <= -5:
            future_5 = df_c.iloc[idx : idx+5]
            p_h5 = train_data['h_pct'].tail(60).quantile(q_h5) * chip_f
            p_l5 = train_data['l_pct'].tail(60).quantile(q_l5) / chip_f
            if future_5['High'].max() >= prev_close * (1 + p_h5): hits["h5"] += 1
            if future_5['Low'].min() <= prev_close * (1 + p_l5): hits["l5"] += 1
            
    return h1, l1, h5, l5, (hits["h1"]/test_days)*100, (hits["l1"]/test_days)*100, (hits["h5"]/test_days)*100, (hits["l5"]/test_days)*100

# --- 介面呈現 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 深度全方位預測系統")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 進入盤中監控模式", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 進入深度分析模式", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    st.title("📊 深度最高/最低價預判")
    if st.button("🏠 回到首頁"): navigate_to("home")
    
    sid = st.text_input("請輸入台股代碼 (如 2330):", key="fc_input")
    if sid:
        with st.spinner("正在進行 AI 運算..."):
            df = yf.download(f"{sid}.TW", period="250d", progress=False)
            if df.empty: df = yf.download(f"{sid}.TWO", period="250d", progress=False)
            
            if not df.empty:
                curr_c = float(df['Close'].iloc[-1])
                h1, l1, h5, l5, ah1, al1, ah5, al5 = ai_full_engine(df)

                # 顯示最新數據
                st.subheader(f"🏠 {sid} 實戰分析報告 (收盤價: {curr_c:.2f})")
                
                # 隔日數據盒
                st.markdown("### 📅 隔日 (T+1) 預測")
                a, b = st.columns(2)
                a.markdown(f"<div style='background:#fff5f5; padding:20px; border-radius:10px;'><h4>📈 預估最高</h4><h2 style='color:red;'>{curr_c*(1+h1):.2f}</h2><p>準確率: {ah1:.1f}%</p></div>", unsafe_allow_html=True)
                b.markdown(f"<div style='background:#f6fff6; padding:20px; border-radius:10px;'><h4>📉 預估最低</h4><h2 style='color:green;'>{curr_c*(1+l1):.2f}</h2><p>準確率: {al1:.1f}%</p></div>", unsafe_allow_html=True)

                # 五日數據盒
                st.markdown("### 🚩 五日 (T+5) 預測")
                c, d = st.columns(2)
                c.markdown(f"<div style='background:#f0f7ff; padding:20px; border-radius:10px;'><h4>🚀 五日最高</h4><h2 style='color:blue;'>{curr_c*(1+h5):.2f}</h2><p>準確率: {ah5:.1f}%</p></div>", unsafe_allow_html=True)
                d.markdown(f"<div style='background:#fffdf0; padding:20px; border-radius:10px;'><h4>⚓ 五日最低</h4><h2 style='color:orange;'>{curr_c*(1+l5):.2f}</h2><p>準確率: {al5:.1f}%</p></div>", unsafe_allow_html=True)

                # 彩色價量圖
                st.divider()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                p_df = df.tail(40).copy()
                ax1.plot(p_df.index, p_df['Close'], color="#1f77b4", label="Price")
                ax1.axhline(curr_c*(1+h1), color='red', ls='--', label="T+1 High")
                ax1.axhline(curr_c*(1+l1), color='green', ls='--', label="T+1 Low")
                ax1.legend()
                
                # 彩色量 (漲紅跌綠)
                colors = ['#e63946' if p_df['Close'].iloc[i] >= p_df['Close'].iloc[i-1] else '#2a9d8f' for i in range(len(p_df))]
                ax2.bar(p_df.index, p_df['Volume'], color=colors)
                st.pyplot(fig)
            else:
                st.error("代碼錯誤或無數據，請重新輸入。")
