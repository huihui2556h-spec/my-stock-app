import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import pytz

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🌍 國際局勢：獲取美股 S&P 500 表現 (保留) ---
def get_international_bias():
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 核心準確率計算函數 (60 日高精度 - 保留) ---
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

# --- 獲取中文名稱 ---
def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 🎨 視覺配色組件 (保留) ---
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
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("整合：國際局勢連動、量能籌碼修正、60日高精度回測、當沖策略指引")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中 AI 即時當沖偵測")
    stock_id = st.text_input("輸入代碼 (例如: 8112):", key="realtime_id")
    if stock_id:
        suffix = ".TW" if len(stock_id) <= 4 else ".TWO"
        df_rt = yf.download(f"{stock_id}{suffix}", period="1d", interval="1m", progress=False)
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            curr_p = float(df_rt['Close'].iloc[-1])
            vwap_p = float(df_rt['VWAP'].iloc[-1])
            st.subheader(f"🎯 {get_stock_name(stock_id)} 盤中分析")
            m1, m2 = st.columns(2)
            m1.metric("即時現價", f"{curr_p:.2f}")
            m2.metric("當前均價 (VWAP)", f"{vwap_p:.2f}")
            st.divider()
            st.markdown("### 🏹 當前分鐘級建議")
            c1, c2 = st.columns(2)
            c1.success(f"🔹 建議買進價位：{vwap_p * 1.001:.2f}")
            c2.error(f"🔸 建議賣出價位：{curr_p * 1.015:.2f}")
        else:
            st.warning("目前抓取不到盤中數據，請確認是否為開盤時間。")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 深度預估分析")
    stock_id = st.text_input("輸入代碼 (例如: 2330):", key="forecast_id")
    if stock_id:
        suffix = ".TW" if len(stock_id) <= 4 else ".TWO"
        df = yf.download(f"{stock_id}{suffix}", period="150d", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.ffill()
            name = get_stock_name(stock_id)
            market_bias, market_pct = get_international_bias()
            vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
            vol_factor = 1.05 if df['Volume'].iloc[-1] > vol_ma5 else 0.95 
            close = df['Close']
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            curr_c = float(close.iloc[-1])
            est_open = curr_c + (atr * 0.05 * market_bias)
            
            # 準確率與預估計算
            acc_h1 = calculate_real_accuracy(df, 0.85, 'high')
            acc_h5 = calculate_real_accuracy(df, 1.9, 'high')
            acc_l1 = calculate_real_accuracy(df, 0.65, 'low')
            acc_l5 = calculate_real_accuracy(df, 1.6, 'low')
            pred_h1 = curr_c + (atr * 0.85 * market_bias * vol_factor)
            pred_h5 = curr_c + (atr * 1.9 * market_bias * vol_factor)
            pred_l1 = curr_c - (atr * 0.65 / (market_bias * vol_factor))
            pred_l5 = curr_c - (atr * 1.6 / (market_bias * vol_factor))

            st.subheader(f"🏠 {name} ({stock_id})")
            st.write(f"🌍 **國際局勢參考 (S&P 500)**: {market_pct:+.2f}%")
            col1, col2 = st.columns(2)
            col1.metric("收盤價", f"{curr_c:.2f}")
            col2.metric("預估開盤", f"{est_open:.2f}", f"{est_open-curr_c:+.2f}")
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                stock_box("📈 隔日最高", pred_h1, ((pred_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高", pred_h5, ((pred_h5/curr_c)-1)*100, acc_h5, "red")
            with c2:
                stock_box("📉 隔日最低", pred_l1, ((pred_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低", pred_l5, ((pred_l5/curr_c)-1)*100, acc_l5, "green")
            st.divider()
            st.markdown("### 🏹 明日當沖建議價格")
            d1, d2, d3 = st.columns(3)
            d1.info(f"🔹 強勢追多\n\n{est_open - (atr * 0.1 * vol_factor):.2f}")
            d2.error(f"🔹 低接買點\n\n{curr_c - (atr * 0.45 / market_bias):.2f}")
            d3.success(f"🔸 短線獲利\n\n{curr_c + (atr * 0.75 * market_bias):.2f}")
            
            # 圖表繪製
            plot_df = df.tail(40)
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', label="Price")
            ax1.axhline(y=pred_h5, color='red', ls='--')
            ax1.axhline(y=pred_l5, color='green', ls='--')
            st.pyplot(fig)
            st.info("📘 圖表說明：顯示收盤價走勢與 AI 壓力支撐線。")
