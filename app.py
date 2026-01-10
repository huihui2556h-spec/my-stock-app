import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

# 1. 頁面優化
st.set_page_config(page_title="台股預測助手", layout="centered")

def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    name, is_disposed = f"股票 {sid}", False
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search:
            name = title_search.group(1).split('-')[0].strip()
        if "處置" in res.text:
            is_disposed = True
    except: pass
    return name, is_disposed

st.title("📈 台股精準預測 APP")
stock_id = st.text_input("輸入股票代碼 (如 8088):", value="8088")

if stock_id:
    ticker_str = f"{stock_id}.TWO" if int(stock_id) > 1000 else f"{stock_id}.TW"
    df = yf.download(ticker_str, period="60d", progress=False, auto_adjust=True)
    
    if not df.empty:
        # 數據處理
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close = df['Close'].ffill()
        high = df['High'].ffill()
        low = df['Low'].ffill()
        volume = df['Volume'].ffill()
        
        stock_name, is_disposed = get_clean_info(stock_id)
        
        # 2. 核心預測與「漲跌幅」計算
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean()
        tp = (high + low + close) / 3
        mf_flow = np.where(tp > tp.shift(1), tp * volume, -tp * volume)
        
        adj = 0.65 if is_disposed else 1.0
        today_close = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        
        mf_strength = np.clip(pd.Series(mf_flow).tail(5).mean() / (pd.Series(mf_flow).tail(20).std() + 1e-9), -1, 1)
        pred_next = today_close + (atr_val * (0.7 + mf_strength * 0.3) * adj)
        pred_5day = today_close + (atr_val * (1.6 + mf_strength * 0.5) * adj)

        # 計算百分比（準確率參考）
        diff_next = ((pred_next / today_close) - 1) * 100
        diff_5day = ((pred_5day / today_close) - 1) * 100

        # 3. 介面顯示 (確保百分比出現)
        st.subheader(f"📊 {stock_name} ({stock_id})")
        st.metric("今日收盤價", f"{today_close:.2f}")
        
        col1, col2 = st.columns(2)
        # 這裡會顯示預估的漲幅百分比
        col1.metric("預估隔日最高", f"{pred_next:.2f}", f"{diff_next:+.2f}%")
        col2.metric("預估五日最高", f"{pred_5day:.2f}", f"{diff_5day:+.2f}%")

        # 4. 繪圖 (使用英文標籤避開亂碼，但在 Streamlit 用文字說明)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(df.index, close, color='#1f77b4', linewidth=2, label="Price (收盤價)")
        ax1.scatter(df.index[-1], pred_next, color='orange', s=100, label="Next Day (隔日預測)")
        ax1.scatter(df.index[-1], pred_5day, color='red', marker='*', s=200, label="5-Day (五日預測)")
        
        # 圖片標題改用英文避開口口口，但在網頁上加中文說明
        ax1.set_title(f"{stock_id} Price Trend & Forecast", fontsize=16)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.write("💡 **圖表說明**：藍線為收盤走勢，橘點為隔日預測，紅星為五日預測目標。")
    else:
        st.error("查無資料")
