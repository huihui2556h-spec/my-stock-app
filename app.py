import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心函式：準確率計算 ---
def calculate_real_accuracy(df, atr_factor, side='high'):
    try:
        df_copy = df.copy().ffill()
        backtest_days = min(len(df_copy) - 15, 60)
        if backtest_days <= 0: return 0.0
        hits = 0
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = (df_copy['High'] - df_copy['Low']).rolling(14).mean().iloc[idx-1]
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            pred_val = prev_close + (prev_atr * atr_factor) if side == 'high' else prev_close - (prev_atr * atr_factor)
            if side == 'high' and actual_val <= pred_val: hits += 1
            elif side == 'low' and actual_val >= pred_val: hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"股票 {stock_id}"

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

# --- 主程式 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價分析")
    stock_id = st.text_input("輸入代碼:", key="rt_id")
    if stock_id:
        symbol = f"{stock_id}.TW"
        df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        if df_rt.empty:
            df_rt = yf.download(f"{stock_id}.TWO", period="1d", interval="1m", progress=False)
        
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            curr_p = float(df_rt['Close'].iloc[-1])
            vwap_p = float(df_rt['VWAP'].iloc[-1])
            st.subheader(f"🎯 {get_stock_name(stock_id)}")
            st.metric("即時現價", f"{curr_p:.2f}")
            c1, c2 = st.columns(2)
            c1.success(f"🔹 建議買進價：{vwap_p * 1.001:.2f}")
            c2.error(f"🔸 建議賣出價：{curr_p * 1.015:.2f}")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 深度預估與波段分析")
    stock_id = st.text_input("輸入代碼:", key="fc_id")
    if stock_id:
        symbol = f"{stock_id}.TW"
        df = yf.download(symbol, period="100d", progress=False)
        if df.empty: df = yf.download(f"{stock_id}.TWO", period="100d", progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.ffill()
            close, high, low = df['Close'], df['High'], df['Low']
            atr = (high - low).rolling(14).mean().iloc[-1]
            curr_c = float(close.iloc[-1])
            
            # 預估值
            p_h1, p_h5 = curr_c + atr*0.85, curr_c + atr*1.9
            p_l1, p_l5 = curr_c - atr*0.65, curr_c - atr*1.6
            acc_h1 = calculate_real_accuracy(df, 0.85, 'high')
            acc_h5 = calculate_real_accuracy(df, 1.9, 'high')
            acc_l1 = calculate_real_accuracy(df, 0.65, 'low')
            acc_l5 = calculate_real_accuracy(df, 1.6, 'low')

            st.subheader(f"🏠 {get_stock_name(stock_id)}")
            col1, col2 = st.columns(2)
            with col1:
                stock_box("📈 隔日最高預測", p_h1, ((p_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高預測", p_h5, ((p_h5/curr_c)-1)*100, acc_h5, "red")
            with col2:
                stock_box("📉 隔日最低預測", p_l1, ((p_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低預測", p_l5, ((p_l5/curr_c)-1)*100, acc_l5, "green")

            # --- 實戰建議文字 ---
            st.divider()
            st.warning("💡 **實戰當沖建議**")
            d1, d2 = st.columns(2)
            d1.write(f"🔹 **多方進場點**：{curr_c - atr*0.1:.2f}")
            d1.write(f"🔹 **超跌低接點**：{curr_c - atr*0.4:.2f}")
            d2.write(f"🔸 **短線分批停利**：{curr_c + atr*0.7:.2f}")

            # --- 完整的價量走勢圖表 (Matplotlib) ---
            st.divider()
            st.write("### 📉 走勢與量價動能表")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
            
            # 上圖：收盤價與 AI 壓力支撐線
            ax1.plot(df.index[-40:], close.tail(40), color='#1f77b4', lw=2, label="Price")
            ax1.axhline(y=p_h5, color='red', ls='--', alpha=0.3, label="Resistance")
            ax1.axhline(y=p_l5, color='green', ls='--', alpha=0.3, label="Support")
            ax1.set_title("Price Analysis", fontsize=14)
            ax1.legend(loc='upper left')
            
            # 下圖：量價表（紅漲綠跌）
            colors = ['red' if close.iloc[i] >= close.iloc[i-1] else 'green' for i in range(-40, 0)]
            ax2.bar(df.index[-40:], df['Volume'].tail(40), color=colors, alpha=0.6)
            ax2.set_title("Volume Momentum", fontsize=12)
            
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # --- 底部註解敘述 ---
            st.info("📘 **圖表與數據說明**")
            st.markdown("""
            * **紅綠量價表**：下方柱狀圖紅色代表收紅K（量增強勢），綠色代表收黑K（量縮整理）。
            * **AI 達成率**：基於過去 60 天波動率對預測價位的命中統計。
            * **實戰操作**：若開盤價即跌破「多方進場點」，代表當日盤勢極弱，不建議進場。
            """)
