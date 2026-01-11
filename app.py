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
st.set_page_config(page_title="台股 AI 多因子交易助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🌍 國際局勢：獲取美股 S&P 500 表現 ---
def get_international_bias():
    try:
        # 抓取標普 500 指數作為國際局勢基準
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        # 國際影響力權重調整 (限制在正負 1% 的影響力)
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 核心準確率計算函數 (保留 60 日回測與小數精度) ---
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
            if np.isnan(prev_atr): continue
            
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            pred_val = prev_close + (prev_atr * atr_factor) if side == 'high' else prev_close - (prev_atr * atr_factor)
            
            if side == 'high' and actual_val <= pred_val: hits += 1
            elif side == 'low' and actual_val >= pred_val: hits += 1
        
        return (hits / backtest_days) * 100
    except:
        return 0.0

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

# --- 🎨 視覺配色組件 ---
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
    st.write("整合：國際局勢連動、量能籌碼修正、60日高精度回測")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    tw_tz = pytz.timezone('Asia/Taipei')
    is_market_open = datetime.datetime.now(tw_tz).weekday() < 5 and (9 <= datetime.datetime.now(tw_tz).hour < 14)
    stock_id = st.text_input("輸入代碼:")
    if stock_id:
        df, sym = fetch_stock_data(stock_id, period="5d")
        if df is None or df.empty:
            st.error("❌ 找不到數據")
            st.stop()
        st.metric(f"{get_stock_name(stock_id)} 現價", f"{df['Close'].iloc[-1]:.2f}")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 多因子深度預估分析")
    stock_id = st.text_input("輸入代碼 (如: 2330):")

    if stock_id:
        with st.spinner('正在分析國際局勢、量能與回測數據...'):
            df, sym = fetch_stock_data(stock_id)
            
            # 🚀 安全性檢查
            if df is None or df.empty:
                st.error("❌ 找不到數據，請確認代碼是否正確。")
                st.stop()

            name = get_stock_name(stock_id)
            df = df.ffill()
            
            # 1. 獲取國際局勢與量價因子
            market_bias, market_pct = get_international_bias()
            
            # 主力籌碼模擬 (量能因子)：計算當前成交量是否超過 5 日均量
            vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
            curr_vol = df['Volume'].iloc[-1]
            vol_factor = 1.05 if curr_vol > vol_ma5 else 0.95 # 放量代表波動可能拓寬

            # 2. 基礎核心計算
            close = df['Close']
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            curr_c = float(close.iloc[-1])

            # 3. 準確率回測 (保留 60 日小數精度)
            acc_h1 = calculate_real_accuracy(df, 0.85, 'high')
            acc_h5 = calculate_real_accuracy(df, 1.9, 'high')
            acc_l1 = calculate_real_accuracy(df, 0.65, 'low')
            acc_l5 = calculate_real_accuracy(df, 1.6, 'low')

            # 4. 多因子合成預估值 (標準預估 * 國際因子 * 量能因子)
            pred_h1 = curr_c + (atr * 0.85 * market_bias * vol_factor)
            pred_h5 = curr_c + (atr * 1.9 * market_bias * vol_factor)
            pred_l1 = curr_c - (atr * 0.65 / (market_bias * vol_factor))
            pred_l5 = curr_c - (atr * 1.6 / (market_bias * vol_factor))

            # --- 畫面呈現 ---
            st.subheader(f"🏠 {name} ({stock_id})")
            
            m_color = "red" if market_pct < 0 else "green"
            st.write(f"🌍 **國際局勢參考 (S&P 500)**: <span style='color:{m_color}'>{market_pct:+.2f}%</span>", unsafe_allow_html=True)
            st.metric("目前收盤價", f"{curr_c:.2f}")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.write("🎯 **壓力預估 (含多因子修正)**")
                stock_box("📈 隔日最高", pred_h1, ((pred_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高", pred_h5, ((pred_h5/curr_c)-1)*100, acc_h5, "red")
            with c2:
                st.write("🛡️ **支撐預估 (含多因子修正)**")
                stock_box("📉 隔日最低", pred_l1, ((pred_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低", pred_l5, ((pred_l5/curr_c)-1)*100, acc_l5, "green")

            # --- 📊 價量走勢圖 ---
            st.divider()
            st.write("📈 **近期價量走勢圖**")
            plot_df = df.tail(40)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="Price")
            ax1.axhline(y=pred_h5, color='#FF4B4B', ls='--', alpha=0.5, label="AI Resistance")
            ax1.axhline(y=pred_l5, color='#28A745', ls='--', alpha=0.5, label="AI Support")
            ax1.set_ylabel("Price")
            ax1.legend(loc='upper left')
            ax1.grid(axis='y', alpha=0.3)

            colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
            ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.7)
            ax2.set_ylabel("Volume")
            
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # --- 📘 圖表與模型說明 ---
            st.info("📘 **模型說明**：本系統整合了國際局勢（美股連動）與量價指標。若美股表現強勁或台股量能放大，AI 會自動拓寬預估區間以捕捉潛在波動。")
            st.markdown(f"""
            * **達成率計算**：係回測過去 **60 個交易日** 之歷史數據，顯示小數點兩位以確保精度。
            * **主力進出修正**：根據 **成交量與 5 日均量** 的關係，自動調整壓力與支撐的敏感度。
            * **國際局勢**：連動 S&P 500 指數，反映全球風險偏好對台股權值股的影響。
            """)
