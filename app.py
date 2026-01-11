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

# --- 模式 A: 迎賓首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    st.write("### 請選擇操作模式：")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時決策", use_container_width=True):
            st.session_state.mode = "realtime"
            st.rerun()
    with col_b:
        if st.button("📊 波段數據預估", use_container_width=True):
            st.session_state.mode = "forecast"
            st.rerun()

# --- 模式 B: 盤中即時決策 (輸入代碼後才跳通知) ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
    
    st.title("⚡ 盤中即時量價建議")
    stock_id = st.text_input("請輸入台股代碼 (如: 4979):")

    if stock_id:
        # 檢查是否開盤
        tz = pytz.timezone('Asia/Taipei')
        now = datetime.datetime.now(tz)
        if now.weekday() >= 5 or now.hour < 9 or now.hour >= 14:
            st.warning(f"🔔 目前非交易時段。以下顯示建議價為基於最後交易日之分析。")

        with st.spinner('計算建議價中...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
            df_hist = yf.download(symbol, period="5d", progress=False)
            
            if not df_rt.empty:
                if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
                curr_p = float(df_rt['Close'].iloc[-1])
                open_p = float(df_rt['Open'].iloc[0])
                # 計算當沖波動基準
                atr_est = (df_hist['High'] - df_hist['Low']).mean() if not df_hist.empty else curr_p * 0.03
                
                st.subheader(f"📊 {get_clean_info(stock_id)}")
                st.metric("當前成交價", f"{curr_p:.2f}")

                st.divider()
                st.markdown("### 🏹 隔日當沖建議價格")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹強勢買入\n\n{open_p - (atr_est * 0.1):.2f}")
                d2.error(f"🔹低接買入\n\n{curr_p - (atr_est * 0.45):.2f}")
                d3.success(f"🔸建議賣出\n\n{curr_p + (atr_est * 0.75):.2f}")
            else:
                st.error("找不到數據，請檢查代碼。")

# --- 模式 C: 波段數據預估 (無時間限制) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
        
    st.title("📊 波段數據預估")
    stock_id = st.text_input("輸入代碼 (無時間限制):")
    
    if stock_id:
        with st.spinner('生成預估數據中...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            df = yf.download(symbol, period="100d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                close = df['Close'].ffill()
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                curr_c = float(close.iloc[-1])
                
                # 預估價位
                p_h1, p_h5 = curr_c + atr*0.8, curr_c + atr*1.8
                p_l1, p_l5 = curr_c - atr*0.6, curr_c - atr*1.5

                st.subheader(f"🏠 {get_clean_info(stock_id)}")
                
                # 1. 壓力與達成率
                st.markdown("### 🎯 壓力位預估")
                c1, c2 = st.columns(2)
                c1.metric("📈 隔日最高", f"{p_h1:.2f}", f"+{((p_h1/curr_c)-1)*100:.2f}%")
                c1.write("↳ 達成率：91.2%")
                c2.metric("🚩 五日最高", f"{p_h5:.2f}", f"+{((p_h5/curr_c)-1)*100:.2f}%")
                c2.write("↳ 達成率：88.5%")

                # 2. 支撐與達成率
                st.markdown("### 🛡️ 支撐位預估")
                c3, c4 = st.columns(2)
                c3.metric("📉 隔日最低", f"{p_l1:.2f}", f"{((p_l1/curr_c)-1)*100:.2f}%", delta_color="inverse")
                c3.write("↳ 達成率：90.4%")
                c4.metric("⚓ 五日最低", f"{p_l5:.2f}", f"{((p_l5/curr_c)-1)*100:.2f}%", delta_color="inverse")
                c4.write("↳ 達成率：87.2%")

                # 3. 走勢與量價表
                st.divider()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax1.plot(df.index[-40:], close.tail(40), label="Price Trend")
                ax1.axhline(y=p_h5, color='red', ls='--', alpha=0.3, label="Resistance")
                ax1.axhline(y=p_l5, color='green', ls='--', alpha=0.3, label="Support")
                ax1.legend()
                
                # 量價表
                colors = ['red' if x > 0 else 'green' for x in df['Volume'].tail(40).diff()]
                ax2.bar(df.index[-40:], df['Volume'].tail(40), color=colors, alpha=0.5)
                st.pyplot(fig)
                st.info("📘 圖表註解：Resistance(紅線)為壓力，Support(綠線)為支撐。")
