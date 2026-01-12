import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# --- 0. 徹底解決圖片與介面亂碼 ---
matplotlib.rc('font', family='Microsoft JhengHei' if 'Win' in str(matplotlib.get_backend()) else 'sans-serif')
plt.rcParams['axes.unicode_minus'] = False 

st.set_page_config(page_title="AI 全景預估系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心功能：60日真實回測命中率 (判斷基準) ---
def calculate_accuracy(df, factor, side='high'):
    try:
        temp_df = df.copy().ffill()
        lookback = 60 
        if len(temp_df) < lookback + 15: return 0.0
        hits, total_days = 0, 0
        for i in range(len(temp_df) - lookback, len(temp_df)):
            history = temp_df.iloc[:i]
            actual_high, actual_low = temp_df['High'].iloc[i], temp_df['Low'].iloc[i]
            prev_close = temp_df['Close'].iloc[i-1]
            tr = np.maximum(history['High'] - history['Low'], 
                           np.maximum(abs(history['High'] - history['Close'].shift(1)), 
                                      abs(history['Low'] - history['Close'].shift(1))))
            current_atr = tr.rolling(14).mean().iloc[-1]
            if np.isnan(current_atr): continue
            total_days += 1
            if side == 'high' and actual_high <= (prev_close + (current_atr * factor)): hits += 1
            elif side == 'low' and actual_low >= (prev_close - (current_atr * factor)): hits += 1
        return (hits / total_days * 100) if total_days > 0 else 0.0
    except: return 0.0

# --- 🔍 數據抓取與名稱辨識 ---
def get_stock_info(stock_id):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period="150d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
                res = requests.get(url, timeout=5)
                name = re.search(r'<title>(.*?) \(', res.text).group(1).split('-')[0].strip()
                return df, symbol, name
        except: continue
    return None, None, f"台股 {stock_id}"

# --- 🎨 還原圖片排版 UI (無亂碼版) ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; border: 1px solid #eee; text-align: center;">
            <p style="margin:0; font-size:16px; color:#666;">{title}</p>
            <h2 style="margin:0; padding:5px 0; color:{text_color}; font-size:32px;">{price:.2f}</h2>
            <p style="margin:0; font-size:12px; color:#888;">60日回測命中率: {accuracy:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式控制 ---
if st.session_state.mode == "home":
    st.title("⚖️ AI 多因子預估分析系統")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 隔日當沖與波段預估", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    now = datetime.now()
    is_trading = (now.weekday() < 5) and (9 <= now.hour < 14)
    if not is_trading:
        st.warning("🕒 【未開盤警示】目前非台股交易時段，顯示為前一交易日行情。")
    sid = st.text_input("輸入股票代碼:")
    if sid:
        df, sym, name = get_stock_info(sid)
        if df is not None:
            st.markdown(f"### {name} ({sym})")
            st.metric("最新成交價", f"{df['Close'].iloc[-1]:.2f}")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日當沖與波段預估")
    stock_input = st.text_input("輸入代碼 (例: 2330):")

    if stock_input:
        with st.spinner('執行 AI 籌碼修正與命中率回測...'):
            df, sym, name = get_stock_info(stock_input)
            if df is not None:
                tr = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
                chip_score = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                bias = 1.006 if chip_score > 1 else 0.994
                curr_p = float(df['Close'].iloc[-1])

                acc_dh = calculate_accuracy(df, (0.85*bias), 'high')
                acc_dl = calculate_accuracy(df, (0.75/bias), 'low')
                acc_wh = calculate_accuracy(df, (1.9*bias), 'high')
                acc_wl = calculate_accuracy(df, (1.6/bias), 'low')

                # 1. 頂部核心：還原名稱與巨型收盤價
                st.divider()
                h1, h2 = st.columns([3, 2])
                with h1:
                    st.markdown(f"<h1 style='color:#000; font-size:55px; margin-bottom:0;'>{name} ({sym})</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background:#f9f9f9; padding:15px; border-radius:10px; border-left:10px solid #C53030; margin-top:10px;'>"
                                f"<p style='color:#444; font-size:22px; margin:0;'>今日收盤價：</p>"
                                f"<b style='font-size:85px; color:#C53030; line-height:1;'>{curr_p:.2f}</b></div>", unsafe_allow_html=True)
                with h2:
                    st.info(f"📊 籌碼修正：{bias:.3f} | 慣性：{(df['Close'].pct_change().std()*100):.2f}\n\n🌅 預估明日開盤：{curr_p + (atr*0.05*bias):.2f}")

                # 2. 全景對照 (還原圖片四欄並排)
                st.markdown("### 🎯 隔日與五日預估對照 (含 60 日真實回測)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: display_metric_card("📈 隔日壓力", curr_p + (atr*0.85*bias), acc_dh, "red")
                with m2: display_metric_card("📉 隔日支撐", curr_p - (atr*0.75/bias), acc_dl, "green")
                with m3: display_metric_card("🚩 五日壓力", curr_p + (atr*1.9*bias), acc_wh, "red")
                with m4: display_metric_card("⚓ 五日支撐", curr_p - (atr*1.6/bias), acc_wl, "green")

                # 3. 當沖建議價格 (還原圖片顏色方塊)
                st.divider()
                st.markdown("### 🏹 明日當沖建議價格")
                st.markdown(f"""
                <div style="display: flex; gap: 15px;">
                    <div style="flex:1; background:#EBF8FF; padding:20px; border-radius:10px; border: 1px solid #BEE3F8; text-align:center;">
                        <b style="color:#2C5282; font-size:20px;">🔹 強勢追多</b><br><span style="font-size:38px; font-weight:bold; color:#2B6CB0;">{curr_p + (atr*0.1):.2f}</span>
                    </div>
                    <div style="flex:1; background:#FFF5F5; padding:20px; border-radius:10px; border: 1px solid #FED7D7; text-align:center;">
                        <b style="color:#9B2C2C; font-size:20px;">🔹 低接買點</b><br><span style="font-size:38px; font-weight:bold; color:#C53030;">{curr_p - (atr*0.45):.2f}</span>
                    </div>
                    <div style="flex:1; background:#F0FFF4; padding:20px; border-radius:10px; border: 1px solid #C6F6D5; text-align:center;">
                        <b style="color:#22543D; font-size:20px;">🔸 短線獲利</b><br><span style="font-size:38px; font-weight:bold; color:#38A169;">{curr_p + (atr*0.75):.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 4. 價量走勢圖 (修正註解亂碼)
                st.divider()
                st.markdown(f"### 📈 {name} 走勢圖與 AI 預估區間")
                plot_df = df.tail(45)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=3, label="收盤價")
                ax1.axhline(y=curr_p + (atr*1.9*bias), color='#FF4B4B', ls='--', lw=2.5, label="五日預估壓力")
                ax1.axhline(y=curr_p - (atr*1.6/bias), color='#28A745', ls='--', lw=2.5, label="五日預估支撐")
                ax1.legend(loc='upper left', fontsize=12)
                ax1.grid(alpha=0.3)
                v_colors = ['#EF5350' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else '#26A69A' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.9)
                st.pyplot(fig)
                st.info(f"💡 圖表說明：藍色粗線為收盤價。紅/綠虛線代表 AI 預估之五日極限空間。")
            else:
                st.error("查開不了，請確認代碼。")
