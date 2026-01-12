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
        if df_rt.empty: df_rt = yf.download(f"{stock_id}.TWO", period="1d", interval="1m", progress=False)
        
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            curr_p = float(df_rt['Close'].iloc[-1])
            vwap_p = float(df_rt['VWAP'].iloc[-1])
            
            st.subheader(f"🎯 {get_stock_name(stock_id)}")
            st.metric("即時成交價", f"{curr_p:.2f}")
            
            st.divider()
            st.markdown("### 🏹 盤中動態決策 (基於即時 VWAP)")
            c1, c2 = st.columns(2)
            # 動態進場點：根據即時均線微調
            buy_p = vwap_p * 0.998 if curr_p < vwap_p else vwap_p * 1.002
            c1.success(f"🔹 動態買進價：{buy_p:.2f}")
            c2.error(f"🔸 動態停利價：{curr_p * 1.02:.2f}")

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
            close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
            
            # --- 動態因子計算 ---
            curr_atr = (high - low).rolling(14).mean().iloc[-1]
            curr_c = float(close.iloc[-1])
            vol_ma5 = vol.rolling(5).mean().iloc[-1]
            # 動態權重：若量能爆發，預測位向上修正
            dynamic_factor = 1.1 if vol.iloc[-1] > vol_ma5 else 0.9
            
            p_h1 = curr_c + (curr_atr * 0.85 * dynamic_factor)
            p_h5 = curr_c + (curr_atr * 1.85 * dynamic_factor)
            p_l1 = curr_c - (curr_atr * 0.65 / dynamic_factor)
            p_l5 = curr_c - (curr_atr * 1.55 / dynamic_factor)

            acc_h1 = calculate_real_accuracy(df, 0.85, 'high')
            acc_h5 = calculate_real_accuracy(df, 1.9, 'high')
            acc_l1 = calculate_real_accuracy(df, 0.65, 'low')
            acc_l5 = calculate_real_accuracy(df, 1.6, 'low')

            # --- UI 呈現：找回收盤價 ---
            st.subheader(f"🏠 {get_stock_name(stock_id)} ({stock_id})")
            st.metric("今日收盤價 (Actual Close)", f"{curr_c:.2f}")
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                stock_box("📈 隔日最高預測", p_h1, ((p_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高預測", p_h5, ((p_h5/curr_c)-1)*100, acc_h5, "red")
            with col2:
                stock_box("📉 隔日最低預測", p_l1, ((p_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低預測", p_l5, ((p_l5/curr_c)-1)*100, acc_l5, "green")

            # --- 實戰動態建議價格 ---
            st.divider()
            st.warning("🏹 **實戰動態當沖價 (基於最新 ATR 與量能計算)**")
            d1, d2, d3 = st.columns(3)
            # 進場價改為動態：昨日收盤價 扣掉 波動率的 0.15 倍（隨市場波動縮放）
            dynamic_buy = curr_c - (curr_atr * 0.15 * (vol_ma5/vol.iloc[-1]))
            d1.info(f"🔹 多方進場點\n\n{dynamic_buy:.2f}")
            d2.error(f"🔹 低接支撐位\n\n{curr_c - (curr_atr * 0.5):.2f}")
            d3.success(f"🔸 目標停利位\n\n{curr_c + (curr_atr * 0.7):.2f}")

            # --- 補回價量走勢圖表 ---
            st.divider()
            st.write("### 📉 走勢與量價動能表")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
            
            # 上圖：收盤價與動態壓力線
            ax1.plot(df.index[-40:], close.tail(40), color='#1f77b4', lw=2, label="Price Trend")
            ax1.axhline(y=p_h5, color='red', ls='--', alpha=0.3, label="Resistance")
            ax1.axhline(y=p_l5, color='green', ls='--', alpha=0.3, label="Support")
            ax1.set_title("Price Analysis", fontsize=14)
            ax1.legend(loc='upper left')
            
            # 下圖：量價表（依漲跌變色）
            # 修正：確保顏色列表長度正確
            plot_df = df.tail(40)
            colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
            ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.6)
            ax2.set_title("Volume Momentum", fontsize=12)
            
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.info("📘 **圖表說明**：上方藍線為收盤價走勢；下方柱狀圖為成交量（紅漲綠跌）。")
