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
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🌍 國際局勢：獲取美股 S&P 500 表現 ---
def get_international_bias():
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 核心準確率計算函數 (60 日高精度) ---
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
    st.write("整合：國際局勢連動、量能籌碼修正、60日高精度回測、當沖策略指引")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    tw_tz = pytz.timezone('Asia/Taipei')
    df_rt, sym_rt = None, None
    stock_id = st.text_input("輸入代碼:")
