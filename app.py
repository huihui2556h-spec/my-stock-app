import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime
import pytz

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心函式：準確率計算 (隔日頁面專用) ---
def calculate_real_accuracy(df, atr_factor, side='high'):
    try:
        df_copy = df.copy().ffill()
        backtest_days = min(len(df_copy) - 15, 60)
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

# --- 🎨 視覺組件 (保留你原本最滿意的紅綠 Box) ---
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

# --- 主選單 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 隔日深度預估", use_container_width=True): navigate_to("forecast")

# =========================================================
# ⚡ 分頁：盤中即時 (動態資料 + 手動重整鍵)
# =========================================================
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    
    header_col, refresh_col = st.columns([4, 1])
    with header_col: st.title("⚡ 盤中動態決策")
    with refresh_col:
        st.write("") 
        if st.button("🔄 重整數據", use_container_width=True): st.rerun()

    stock_id = st.text_input("輸入代碼:", key="rt_id")
    if stock_id:
        symbol = f"{stock_id}.TW"
        df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        if df_rt.empty: df_rt = yf.download(f"{stock_id}.TWO", period="1d", interval="1m", progress=False)
        
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            curr_p = float(df_rt['Close'].iloc[-1])
            vwap_p = float(df_rt['VWAP'].iloc[-1])
            now = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%H:%M:%S")
            
            st.subheader(f"🎯 {get_stock_name(stock_id)}")
            st.caption(f"盤中秒級更新時間：{now}")
            st.metric("即時現價", f"{curr_p:.2f}")
            
            st.divider()
            st.markdown("### 🏹 盤中即時動態建議")
            c1, c2 = st.columns(2)
            c1.success(f"🔹 即時支撐買點 (VWAP)：{vwap_p:.2f}")
            c2.error(f"🔸 即時分批停利：{curr_p * 1.015:.2f}")
        else:
            st.warning("查無即時數據，請確認是否為開盤時間。")

# =========================================================
# 📊 分頁：隔日深度預估 (保留原本完整的收盤價、Box、價量表)
# =========================================================
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日當沖策略預判")
    stock_id = st.text_input("輸入代碼:", key="fc_id")
    if stock_id:
        symbol = f"{stock_id}.TW"
        df = yf.download(symbol, period="100d", progress=False)
        if df.empty: df = yf.download(f"{stock_id}.TWO", period="100d", progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.ffill()
            close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
            curr_c = float(close.iloc[-1])
            atr = (high - low).rolling(14).mean().iloc[-1]
            
            # 隔日點位計算 (保留你原本的公式)
            p_h1, p_h5 = curr_c + atr*0.85, curr_c + atr*1.9
            p_l1, p_l5 = curr_c - atr*0.65, curr_c - atr*1.6
            acc_h1 = calculate_real_accuracy(df, 0.85, 'high')
            acc_h5 = calculate_real_accuracy(df, 1.9, 'high')
            acc_l1 = calculate_real_accuracy(df, 0.65, 'low')
            acc_l5 = calculate_real_accuracy(df, 1.6, 'low')

            # --- 畫面呈現 ---
            st.subheader(f"🏠 {get_stock_name(stock_id)} ({stock_id})")
            st.metric("今日最新收盤價 (Actual Close)", f"{curr_c:.2f}")
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                stock_box("📈 隔日最高預測", p_h1, ((p_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高預測", p_h5, ((p_h5/curr_c)-1)*100, acc_h5, "red")
            with col2:
                stock_box("📉 隔日最低預測", p_l1, ((p_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低預測", p_l5, ((p_l5/curr_c)-1)*100, acc_l5, "green")

            # 隔日靜態參考點 (補回文字與公式)
            st.divider()
            st.markdown("### 🏹 隔日當沖參考點位 (開盤前靜態參考)")
            d1, d2, d3 = st.columns(3)
            d1.info(f"🔹 多方進場參考\n\n{curr_c - (atr * 0.15):.2f}")
            d2.error(f"🔹 空方/低接參考\n\n{curr_c - (atr * 0.5):.2f}")
            d3.success(f"🔸 隔日獲利目標\n\n{curr_c + (atr * 0.7):.2f}")

            # 補回價量走勢圖表 (Matplotlib)
            st.divider()
            st.write("### 📉 歷史走勢與量價動能表")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
            ax1.plot(df.index[-40:], close.tail(40), color='#1f77b4', lw=2, label="Price Trend")
            ax1.axhline(y=p_h5, color='red', ls='--', alpha=0.3, label="Resistance")
            ax1.axhline(y=p_l5, color='green', ls='--', alpha=0.3, label="Support")
            ax1.legend()
            
            plot_df = df.tail(40)
            # 紅綠變色量能
            colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
            ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.6)
            st.pyplot(fig)

            # 補回詳細中文說明
            st.info("📘 **圖表與點位說明**")
            st.markdown("""
            * **收盤價**：顯示最新一個交易日的實際成交收盤價。
            * **隔日參考點位**：由 ATR 指標算出的固定參考值，提供您開盤前的心理點位基準。
            * **量價動能**：下圖柱狀圖紅色代表當日收紅K（漲），綠色代表收黑K（跌）。
            """)
