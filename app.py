import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import requests
import re

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 交易助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 AI 動態準確率與偏差計算函數 ---
def calculate_ai_metrics(df, base_factor, side='high'):
    try:
        df_copy = df.copy().ffill()
        backtest_days = min(len(df_copy) - 15, 60)
        if backtest_days <= 0: return 0.0, 1.0
        
        hits = 0
        total_bias = 0
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = (df_copy['High'] - df_copy['Low']).rolling(14).mean().iloc[idx-1]
            if np.isnan(prev_atr): continue
            
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            pred_val = prev_close + (prev_atr * base_factor) if side == 'high' else prev_close - (prev_atr * base_factor)
            
            # 計算準確率
            if side == 'high':
                if actual_val <= pred_val: hits += 1
            else:
                if actual_val >= pred_val: hits += 1
            
            # 計算偏差值 (用於修正明天的預估值)
            total_bias += (actual_val / pred_val)
            
        accuracy = (hits / backtest_days) * 100
        avg_bias = total_bias / backtest_days # 這是 AI 的修正係數
        return accuracy, avg_bias
    except:
        return 0.0, 1.0

# --- 獲取中文名稱 ---
def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 自動偵測機制 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(stock_id, period="150d"):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period=period, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, symbol
        except: continue
    return None, None

# --- 🎨 自定義台股配色組件 ---
def stock_box(label, price, pct, acc, color_type="red"):
    bg_color = "#FF4B4B" if color_type == "red" else "#28A745"
    arrow = "↑" if color_type == "red" else "↓"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {bg_color}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{bg_color}; color:white; padding:2px 8px; border-radius:5px; font-size:14px;">
                {arrow} {pct:.2f}%
            </span>
            <p style="margin-top:10px; font-size:12px; color:#888;">↳ 近 60 日 AI 達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式邏輯 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 隔日當沖及波段預估", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日當沖與波段預估")
    stock_id = st.text_input("輸入代碼 (如: 2330):")

    if stock_id:
        with st.spinner('AI 正在根據歷史準確率修正預估值...'):
            df, sym = fetch_stock_data(stock_id)
            
            # --- 🚀 安全檢查：找不到數據就停止 ---
            if df is None or df.empty:
                st.error("❌ 找不到數據，請確認代碼是否正確。")
                st.stop()

            name = get_stock_name(stock_id)
            df = df.ffill()
            
            # --- AI 核心計算 ---
            close = df['Close']
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            curr_c = float(close.iloc[-1])
            
            # 獲取準確率與 AI 修正係數 (Bias)
            acc_h1, bias_h1 = calculate_ai_metrics(df, 0.85, 'high')
            acc_h5, bias_h5 = calculate_ai_metrics(df, 1.9, 'high')
            acc_l1, bias_l1 = calculate_ai_metrics(df, 0.65, 'low')
            acc_l5, bias_l5 = calculate_ai_metrics(df, 1.6, 'low')

            # --- 預估值連動：將原始預估乘以 AI 修正係數 ---
            pred_h1 = (curr_c + atr * 0.85) * bias_h1
            pred_h5 = (curr_c + atr * 1.9) * bias_h5
            pred_l1 = (curr_c - atr * 0.65) * bias_l1
            pred_l5 = (curr_c - atr * 1.6) * bias_l5

            st.subheader(f"🏠 {name} ({stock_id}) 預估分析")
            st.metric("目前收盤價", f"{curr_c:.2f}")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.write("🎯 **壓力預估 (動態修正)**")
                stock_box("📈 隔日最高", pred_h1, ((pred_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高", pred_h5, ((pred_h5/curr_c)-1)*100, acc_h5, "red")
            with c2:
                st.write("🛡️ **支撐預估 (動態修正)**")
                stock_box("📉 隔日最低", pred_l1, ((pred_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低", pred_l5, ((pred_l5/curr_c)-1)*100, acc_l5, "green")

            # --- 📊 價量走勢圖 ---
            st.divider()
            st.write("📈 **近期價量走勢圖**")
            plot_df = df.tail(40)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="Price")
            ax1.axhline(y=pred_h5, color='#FF4B4B', ls='--', alpha=0.6, label="AI Resistance")
            ax1.axhline(y=pred_l5, color='#28A745', ls='--', alpha=0.6, label="AI Support")
            ax1.set_ylabel("Price")
            ax1.legend(loc='upper left')
            ax1.grid(axis='y', alpha=0.3)

            colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
            ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.7)
            ax2.set_ylabel("Volume")
            
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # --- 📘 圖表說明區 ---
            st.info("📘 **圖表說明**：上方為收盤價走勢與 AI 壓力支撐線；下方為成交量（紅漲綠跌）。")
            st.markdown(f"""
            * **動態修正說明**：目前的預估值已根據過去 60 天的 **AI 偏差率 (Bias)** 進行優化。若該股近期波動加大，AI 會自動拓寬預估區間。
            * **準確率連動**：數值顯示的小數點位反映了歷史 60 筆交易數據回測的精細度。
            """)
