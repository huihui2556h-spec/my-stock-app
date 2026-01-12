import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子動態回測系統 Pro", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🧬 籌碼模組 ---
def get_institutional_chips(stock_id):
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        chip_weight = 1.0 
        msg = "籌碼動向：中性"
        if not inst_df.empty:
            net = inst_df.tail(9)['buy'].sum() - inst_df.tail(9)['sell'].sum()
            if net > 0:
                chip_weight += 0.018
                msg = "✅ 籌碼動向：法人持續買超"
            else:
                chip_weight -= 0.018
                msg = "⚠️ 籌碼動向：法人持續賣出"
        return round(chip_weight, 4), msg
    except:
        return 1.0, "⚠️ FinMind API 連線中..."

# --- 🧠 AI 核心引擎 ---
def ai_forecast_engine(df, chip_f=1.0):
    vol = df['Close'].pct_change().tail(20).std()
    h1_q, l1_q = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    h5_q, l5_q = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    
    df_clean = df.tail(80).copy()
    df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
    df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
    
    return (df_clean['h_pct'].quantile(h1_q) * chip_f, 
            df_clean['l_pct'].quantile(l1_q) / chip_f,
            df_clean['h_pct'].quantile(h5_q) * chip_f,
            df_clean['l_pct'].quantile(l5_q) / chip_f)

# --- 📊 四重獨立回測引擎 ---
def run_full_backtest(df, chip_f):
    test_days = 20
    hist_data = df.tail(test_days + 65)
    hits = {"h1": 0, "l1": 0, "h5": 0, "l5": 0}
    for i in range(test_days):
        train = hist_data.iloc[i : i+60]
        pc = hist_data.iloc[i+60-1]['Close']
        h1, l1, h5, l5 = ai_forecast_engine(train, chip_f)
        
        # 隔日命中
        if hist_data.iloc[i+60]['High'] >= pc * (1+h1): hits["h1"] += 1
        if hist_data.iloc[i+60]['Low'] <= pc * (1+l1): hits["l1"] += 1
        # 五日命中
        if hist_data.iloc[i+60 : i+65]['High'].max() >= pc * (1+h5): hits["h5"] += 1
        if hist_data.iloc[i+60 : i+65]['Low'].min() <= pc * (1+l5): hits["l5"] += 1
    return {k: (v/test_days)*100 for k, v in hits.items()}

# --- 🎨 顯示盒子 ---
def render_target_box(title, price, pct, acc, color="red"):
    border = "#FF4B4B" if color == "red" else "#28A745"
    st.markdown(f"""
        <div style="border-left: 8px solid {border}; background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
            <div style="color:#555; font-size:14px; font-weight:bold;">{title}</div>
            <div style="font-size:32px; font-weight:bold; color:#111;">{price:.2f}</div>
            <div style="display:flex; justify-content:space-between; margin-top:5px;">
                <span style="color:{border};">變動 {pct:.2f}%</span>
                <span style="background:#eee; padding:2px 5px; border-radius:3px;">🎯 準確率 {acc:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# 🚀 頁面主體
# =========================================================
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 動態回測導航")
    if st.button("📊 進入深度分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    sid = st.text_input("輸入台股代碼 (例: 2330):")
    
    if sid:
        with st.spinner('數據計算中...'):
            df = yf.download(f"{sid}.TW", period="200d", progress=False)
            if df.empty: df = yf.download(f"{sid}.TWO", period="200d", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                
                # 取得最新收盤價與日期
                curr_c = float(df['Close'].iloc[-1])
                curr_date = df.index[-1].strftime('%Y-%m-%d')
                
                chip_f, chip_msg = get_institutional_chips(sid)
                h1, l1, h5, l5 = ai_forecast_engine(df, chip_f)
                bt = run_full_backtest(df, chip_f)

                # --- 頂部資訊顯示 ---
                st.subheader(f"🏠 分析標的：{sid} (基準日: {curr_date})")
                
                # 強調顯示收盤價
                c1, c2 = st.columns([1, 2])
                c1.metric("📌 最新收盤價", f"{curr_c:.2f}")
                c2.info(f"🧬 {chip_msg}")

                st.divider()
                
                # --- 四個預估價與準確率 ---
                st.markdown("### 🎯 點位預估與實戰準確率")
                colA, colB = st.columns(2)
                with colA:
                    render_target_box("隔日高點預估 (T+1)", curr_c*(1+h1), h1*100, bt["h1"], "red")
                    render_target_box("五日極限高點 (T+5)", curr_c*(1+h5), h5*100, bt["h5"], "red")
                with colB:
                    render_target_box("隔日低點預估 (T+1)", curr_c*(1+l1), l1*100, bt["l1"], "green")
                    render_target_box("五日極限低點 (T+5)", curr_c*(1+l5), l5*100, bt["l5"], "green")

                # --- 📉 避開亂碼的圖表 ---
                st.divider()
                st.write("### 📈 Price & AI Bands (Visual Guide)")
                fig, ax = plt.subplots(figsize=(10, 4))
                hist_p = df['Close'].tail(40)
                ax.plot(hist_p.index, hist_p, color="#1f77b4", lw=2, label="Close Price")
                
                # 圖內文字全部英文，徹底解決亂碼
                ax.axhline(y=curr_c*(1+h1), color='red', ls='--', alpha=0.5, label="T+1 High")
                ax.axhline(y=curr_c*(1+l1), color='green', ls='--', alpha=0.5, label="T+1 Low")
                ax.fill_between(hist_p.index, curr_c*(1+l1), curr_c*(1+h1), color='gray', alpha=0.1)
                
                ax.legend(loc='upper left')
                ax.set_title(f"Stock {sid} Forecast Analysis")
                st.pyplot(fig)
                
                st.caption("註：圖表標題與標籤採用英文以防止系統顯示亂碼。")
