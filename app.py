import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

# 1. 頁面優化 (手機瀏覽器自動適應)
st.set_page_config(page_title="台股預測助手", layout="centered")

# 2. 修正亂碼：精準抓取中文名稱
def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    name, is_disposed = f"股票 {sid}", False
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        # 使用正則表達式精確提取 <title> 中的公司名，避開後方的腳本原始碼
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
    # 判斷上市或上櫃
    ticker_str = f"{stock_id}.TWO" if int(stock_id) > 1000 else f"{stock_id}.TW"
    df = yf.download(ticker_str, period="60d", progress=False, auto_adjust=True)
    
    if not df.empty:
        # 數據降維處理 (解決維度報錯)
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close = df['Close'].ffill()
        high = df['High'].ffill()
        low = df['Low'].ffill()
        volume = df['Volume'].ffill()
        
        stock_name, is_disposed = get_clean_info(stock_id)
        
        # 3. 核心預測邏輯 (強化準確度)
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean()
        tp = (high + low + close) / 3
        mf_flow = np.where(tp > tp.shift(1), tp * volume, -tp * volume)
        
        adj = 0.65 if is_disposed else 1.0
        today_close = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        
        # 根據資金流強度調整預測權重
        mf_strength = np.clip(pd.Series(mf_flow).tail(5).mean() / (pd.Series(mf_flow).tail(20).std() + 1e-9), -1, 1)
        pred_next = today_close + (atr_val * (0.7 + mf_strength * 0.3) * adj)
        pred_5day = today_close + (atr_val * (1.6 + mf_strength * 0.5) * adj)

        # 4. 手機介面優化顯示
        st.subheader(f"📊 {stock_name} ({stock_id})")
        st.metric("今日收盤價", f"{today_close:.2f}")
        
        col1, col2 = st.columns(2)
        col1.metric("預估隔日最高", f"{pred_next:.2f}", f"{((pred_next/today_close)-1)*100:.1f}%")
        col2.metric("預估五日最高", f"{pred_5day:.2f}", f"{((pred_5day/today_close)-1)*100:.1f}%")

        # 5. 繪圖 (解決方塊亂碼問題)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2.5, 1]})
        
        # 設定通用字體，嘗試避開亂碼
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif'] 
        
        ax1.plot(df.index, close, color='#1f77b4', linewidth=2, label="Price")
        ax1.scatter(df.index[-1], pred_next, color='orange', s=100, label="Next Day")
        ax1.scatter(df.index[-1], pred_5day, color='red', marker='*', s=200, label="5-Day")
        ax1.set_title(f"{stock_id} Trend & Forecast", fontsize=16)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 買賣超能量圖
        colors = ['#ff9999' if x > 0 else '#99ff99' for x in mf_flow]
        ax2.bar(df.index, mf_flow/1e8, color=colors)
        ax2.set_ylabel("Money Flow (100M)")
        
        st.pyplot(fig)
    else:
        st.error("查無資料，請確認代碼是否正確。")
