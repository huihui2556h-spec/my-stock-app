import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import matplotlib.pyplot as plt
import matplotlib

# --- 0. 設置中文字體 (解決圖片亂碼) ---
matplotlib.rc('font', family='Microsoft JhengHei' if 'Win' in str(matplotlib.get_backend()) else 'sans-serif')
plt.rcParams['axes.unicode_minus'] = False # 解決負號亂碼

# 頁面寬度設定
st.set_page_config(page_title="AI 全景預估 Pro", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心功能：真實回測勝率判斷 ---
def calculate_accuracy(df, factor, side='high'):
    try:
        temp_df = df.copy().ffill()
        lookback = 60
        if len(temp_df) < lookback + 15: return 0.0
        hits = 0
        total_days = 0
        for i in range(len(temp_df) - lookback, len(temp_df)):
            history = temp_df.iloc[:i]
            actual_high = temp_df['High'].iloc[i]; actual_low = temp_df['Low'].iloc[i]
            prev_close = temp_df['Close'].iloc[i-1]
            tr = np.maximum(history['High'] - history['Low'], 
                           np.maximum(abs(history['High'] - history['Close'].shift(1)), 
                                      abs(history['Low'] - history['Close'].shift(1))))
            current_atr = tr.rolling(14).mean().iloc[-1]
            if np.isnan(current_atr): continue
            total_days += 1
            if side == 'high':
                if actual_high <= (prev_close + (current_atr * factor)): hits += 1
            else:
                if actual_low >= (prev_close - (current_atr * factor)): hits += 1
        return (hits / total_days * 100) if total_days > 0 else 0.0
    except: return 0.0

# --- 🔍 數據抓取 ---
def fetch_stock_data(stock_id):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period="150d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, symbol
        except: continue
    return None, None

# --- 🎨 介面組件 ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 12px; border: 1px solid #eee; text-align: center;">
            <p style="margin:0; font-size:14px; color:#666;">{title}</p>
            <h2 style="margin:0; padding:8px 0; color:{text_color};">{price:.2f}</h2>
            <p style="margin:0; font-size:12px; color:#999;">回測命中率: <b>{accuracy:.1f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式 ---
if st.session_state.mode == "home":
    st.title("⚖️ AI 多因子預估全景系統")
    c1, c2 = st.columns(2)
    with c1: 
        if st.button("⚡ 進入：盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c2: 
        if st.button("📊 進入：預估全景分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價分析")
    sid = st.text_input("輸入代碼:")
    if sid:
        df, sym = fetch_stock_data(sid)
        if df is not None:
            st.metric(f"最新成交價 ({sym})", f"{df['Close'].iloc[-1]:.2f}")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    stock_input = st.text_input("請輸入分析代碼 (例: 2330):")

    if stock_input:
        with st.spinner('AI 正在同步回測數據...'):
            df, sym = fetch_stock_data(stock_input)
            if df is not None:
                # 數據計算
                tr = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
                chip_score = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                bias = 1.006 if chip_score > 1 else 0.994
                curr_p = float(df['Close'].iloc[-1])

                # 真實回測
                acc_dh = calculate_accuracy(df, (0.85*bias), 'high')
                acc_dl = calculate_accuracy(df, (0.75/bias), 'low')
                acc_wh = calculate_accuracy(df, (1.9*bias), 'high')
                acc_wl = calculate_accuracy(df, (1.6/bias), 'low')

                # 1. 收盤價獨立欄位 (頂部)
                st.divider()
                header_c1, header_c2 = st.columns([2, 3])
                with header_c1:
                    st.markdown(f"<p style='color:#666; font-size:18px; margin-bottom:0;'>{sym} 今日收盤價</p>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='font-size:64px; margin-top:0;'>{curr_p:.2f}</h1>", unsafe_allow_html=True)
                with header_c2:
                    st.info(f"💡 籌碼修正: {bias:.3f} | 法人態度: {'偏多' if bias > 1 else '偏空'}\n\n預估明日開盤: {curr_p + (atr*0.05*bias):.2f}")

                # 2. 隔日與五日整合段落
                st.markdown("### 🎯 核心預估對照 (含 60 日真實回測)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: display_metric_card("📈 隔日壓力", curr_p + (atr*0.85*bias), acc_dh, "red")
                with m2: display_metric_card("📉 隔日支撐", curr_p - (atr*0.75/bias), acc_dl, "green")
                with m3: display_metric_card("🚩 五日最大壓力", curr_p + (atr*1.9*bias), acc_wh, "red")
                with m4: display_metric_card("⚓ 五日最大支撐", curr_p - (atr*1.6/bias), acc_wl, "green")

                # 3. 當沖建議價格
                st.divider()
                st.markdown("### 🏹 明日當沖建議")
                d1, d2, d3 = st.columns(3)
                d1.warning(f"🔹 強勢追多: {curr_p + (atr*0.1):.2f}")
                d2.error(f"🔹 低接買點: {curr_p - (atr*0.45):.2f}")
                d3.success(f"🔸 短線獲利: {curr_p + (atr*0.75):.2f}")

                # 4. 價量圖 (修正亂碼與排版)
                st.divider()
                st.markdown("### 📈 近期價量走勢與 AI 區間")
                plot_df = df.tail(40)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                # 價格與虛線註解
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="收盤價")
                ax1.axhline(y=curr_p + (atr*1.9*bias), color='#FF4B4B', ls='--', alpha=0.5, label="5D 壓力線")
                ax1.axhline(y=curr_p - (atr*1.6/bias), color='#28A745', ls='--', alpha=0.5, label="5D 支撐線")
                ax1.legend(loc='upper left', fontsize=10)
                ax1.set_ylabel("價格", fontsize=12)
                ax1.grid(alpha=0.2)
                
                # 成交量
                colors = ['#EF5350' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else '#26A69A' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.8)
                ax2.set_ylabel("成交量", fontsize=12)
                
                st.pyplot(fig)
                st.caption("📘 圖表註解：紅虛線與綠虛線分別代表 AI 預估之波段極限。")
            else:
                st.error("查無資料")
