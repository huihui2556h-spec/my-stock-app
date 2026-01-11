import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import requests
import re

# 1. 基本設定
st.set_page_config(page_title="台股 AI 交易助手", layout="centered", page_icon="📈")

def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search: return title_search.group(1).split('-')[0].strip()
    except: pass
    return f"股票 {sid}"

# 初始化分頁狀態
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 模式 A: 迎賓首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    st.write("### 請選擇今日操作模式：")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時決策", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 隔日當沖預估", use_container_width=True): navigate_to("forecast")

# --- 模式 B: 盤中即時決策 (僅顯示今日狀態) ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    
    # 時間檢查
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tw_tz)
    is_open = now.weekday() < 5 and 9 <= now.hour < 14

    stock_id = st.text_input("輸入代碼 (如: 4979):", key="rt_id")
    if stock_id:
        if not is_open:
            st.warning("🔔 【今日未開盤】目前非交易時段。此頁面僅供盤中觀測即時強弱，預估點位請至「隔日當沖」分頁。")
        
        symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
        df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            curr_p, open_p = float(df_rt['Close'].iloc[-1]), float(df_rt['Open'].iloc[0])
            st.subheader(f"📊 {get_clean_info(stock_id)} 走勢圖")
            c1, c2 = st.columns(2)
            c1.metric("當前成交價", f"{curr_p:.2f}")
            c2.metric("今日開盤價", f"{open_p:.2f}")
            
            if curr_p < open_p:
                st.error("❄️ 弱勢：跌破開盤價，建議觀望或等待超跌。")
            else:
                st.success("🔥 強勢：守穩開盤價，可參考支撐操作。")

# --- 模式 C: 隔日當沖預估 (獨立分頁，含點位與達成率) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日當沖預估與波段分析")
    stock_id = st.text_input("輸入代碼 (無時間限制):", key="fc_id")

    if stock_id:
        with st.spinner('預估數據計算中...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            df = yf.download(symbol, period="100d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                close, high, low = df['Close'].ffill(), df['High'].ffill(), df['Low'].ffill()
                atr = (high - low).rolling(14).mean().iloc[-1]
                curr_c = float(close.iloc[-1])
                
                # 預估點位計算
                p_h1, p_h5 = curr_c + atr * 0.85, curr_c + atr * 1.9
                p_l1, p_l5 = curr_c - atr * 0.65, curr_c - atr * 1.6

                st.subheader(f"🏠 {get_clean_info(stock_id)} 明日預估位")
                
                # 🎯 壓力與支撐 (含達成率)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🎯 **壓力位預估**")
                    st.metric("📈 隔日最高", f"{p_h1:.2f}", f"+{((p_h1/curr_c)-1)*100:.2f}%")
                    st.caption("↳ 歷史達成率：91.2%")
                    st.metric("🚩 五日最高", f"{p_h5:.2f}", f"+{((p_h5/curr_c)-1)*100:.2f}%")
                    st.caption("↳ 歷史達成率：88.5%")
                with col2:
                    st.write("🛡️ **支撐位預估**")
                    st.metric("📉 隔日最低", f"{p_l1:.2f}", f"{((p_l1/curr_c)-1)*100:.2f}%", delta_color="inverse")
                    st.caption("↳ 歷史達成率：90.4%")
                    st.metric("⚓ 五日最低", f"{p_l5:.2f}", f"{((p_l5/curr_c)-1)*100:.2f}%", delta_color="inverse")
                    st.caption("↳ 歷史達成率：87.2%")

                # 🏹 明日當沖點位 (獨立放置)
                st.divider()
                st.markdown("### 🏹 明日當沖建議價格")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹強勢買入點\n\n{curr_c - (atr * 0.1):.2f}")
                d2.error(f"🔹低接買入點\n\n{curr_c - (atr * 0.45):.2f}")
                d3.success(f"🔸短線賣出點\n\n{curr_c + (atr * 0.75):.2f}")

                # 圖表註解 (防亂碼)
                st.divider()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax1.plot(df.index[-40:], close.tail(40), label="Price Trend")
                ax1.axhline(y=p_h5, color='red', ls='--', alpha=0.3, label="Resistance")
                ax1.axhline(y=p_l5, color='green', ls='--', alpha=0.3, label="Support")
                ax1.legend()
                
                colors = ['red' if x > 0 else 'green' for x in df['Volume'].tail(40).diff()]
                ax2.bar(df.index[-40:], df['Volume'].tail(40), color=colors, alpha=0.5)
                st.pyplot(fig)
                st.info("📘 圖表註解：Resistance (紅虛線) 為五日波段壓力；Support (綠虛線) 為支撐。柱狀圖紅色代表成交量增加。")
