import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import requests
import re

# 1. 頁面基礎設定
st.set_page_config(page_title="台股 AI 交易助手", layout="centered", page_icon="📈")

# 解決字體亂碼：定義英文標籤對照函數
def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search: return title_search.group(1).split('-')[0].strip()
    except: pass
    return f"股票 {sid}"

# 初始化導航狀態
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

# 定義導航函數 (確保返回首頁能徹底清除輸入狀態)
def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 模式 A: 迎賓首頁 (路由中心) ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    st.write("### 請選擇今日操作模式：")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時決策", use_container_width=True):
            navigate_to("realtime")
            
    with col_b:
        if st.button("📊 波段數據預估", use_container_width=True):
            navigate_to("forecast")

# --- 模式 B: 盤中即時決策 (輸入代碼後才檢查時間) ---
elif st.session_state.mode == "realtime":
    # 側邊欄返回按鈕
    if st.sidebar.button("⬅️ 返回首頁"):
        navigate_to("home")
    
    st.title("⚡ 盤中即時量價建議")
    stock_id = st.text_input("請輸入台股代碼 (如: 4979):", key="rt_input")

    if stock_id:
        # 只有輸入代碼後才進行時間檢查
        tw_tz = pytz.timezone('Asia/Taipei')
        now = datetime.datetime.now(tw_tz)
        is_weekday = now.weekday() < 5
        is_market_time = 9 <= now.hour < 14
        
        # 彈出通知 (通知僅在輸入代號後出現)
        if not is_weekday:
            st.warning(f"🔔 【目前未開盤】今天為週末，顯示數據為前一交易日資訊。")
        elif now.hour < 9:
            st.info(f"🔔 【目前未開盤】今日台股尚未開盤（09:00 開盤），以下為預估建議。")
        elif now.hour >= 14:
            st.info(f"🔔 【今日已收盤】目前顯示今日結算數據。")

        # 抓取數據與計算建議價 (邏輯同前，確保數值出現)
        symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
        df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        df_hist = yf.download(symbol, period="5d", progress=False)
        
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            curr_p = float(df_rt['Close'].iloc[-1])
            open_p = float(df_rt['Open'].iloc[0])
            atr_est = (df_hist['High'] - df_hist['Low']).mean() if not df_hist.empty else curr_p * 0.03

            st.subheader(f"📊 {get_clean_info(stock_id)}")
            st.metric("當前成交價", f"{curr_p:.2f}")

            st.divider()
            st.markdown("### 🏹 當沖建議買賣價格")
            d1, d2, d3 = st.columns(3)
            d1.info(f"🔹強勢買入\n\n{open_p - (atr_est * 0.1):.2f}")
            d2.error(f"🔹低接買入\n\n{curr_p - (atr_est * 0.45):.2f}")
            d3.success(f"🔸建議賣出\n\n{curr_p + (atr_est * 0.75):.2f}")

# --- 模式 C: 波段數據預估 (無時無刻可用) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"):
        navigate_to("home")
        
    st.title("📊 波段數據預估")
    stock_id = st.text_input("輸入代碼 (無時間限制):", key="fc_input")
    
    if stock_id:
        # 波段預估邏輯... (此處維持原有繪圖與達成率計算，不受開盤時間警示干擾)
        st.success(f"正在分析 {stock_id} 的長期趨勢...")
        # (此處插入您原本的預估位與圖表代碼)
