import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

# 頁面配置
st.set_page_config(page_title="台股預測助手", layout="centered")

# 精準抓取中文名稱，排除原始碼
def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    name, is_disposed = f"股票 {sid}", False
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        # 只抓取標題中第一個括號前的純文字
        title_match = re.search(r'<title>(.*?) \(', res.text)
        if title_match:
            name = title_match.group(1).strip()
        if "處置" in res.text:
            is_disposed = True
    except: pass
    return name, is_disposed

st.title("📈 台股精準預測 APP")
stock_id = st.text_input("輸入股票代碼:", value="8088")

if stock_id:
    ticker_str = f"{stock_id}.TWO" if int(stock_id) > 1000 else f"{stock_id}.TW"
    df = yf.download(ticker_str, period="60d", progress=False, auto_adjust=True)
    
    if not df.empty:
        # 強制轉為一維數據避免報錯
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        stock_name, is_disposed = get_clean_info(stock_id)
        
        # 核心計算
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean()
        tp = (high + low + close) / 3
        mf_flow = np.where(tp > tp.shift(1), tp * volume, -tp * volume)
        
        adj = 0.65 if is_disposed else 1.0
        today_close = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        
        pred_next = today_close + (atr_val * 0.7 * adj)
        pred_5day = today_close + (atr_val * 1.6 * adj)

        # 手機版大卡片顯示
        st.subheader(f"📊 {stock_name} ({stock_id})")
        st.metric("今日收盤價", f"{today_close:.2f}")
        c1, c2 = st.columns(2)
        c1.metric("預估隔日最高", f"{pred_next:.2f}")
        c2.metric("預估五日最高", f"{pred_5day:.2f}")

        # 繪圖 (買賣超量絕對不會不見)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2.5, 1]})
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        
        ax1.plot(df.index, close, color='#1f77b4', linewidth=2.5)
        ax1.scatter(df.index[-1], pred_next, color='orange', s=150)
        ax1.scatter(df.index[-1], pred_5day, color='red', marker='*', s=250)
        ax1.set_title("價格趨勢與預測")
        
        colors = ['red' if x > 0 else 'green' for x in mf_flow]
        ax2.bar(df.index, mf_flow/1e8, color=colors)
        ax2.set_ylabel("資金流向 (億)")
        
        st.pyplot(fig)