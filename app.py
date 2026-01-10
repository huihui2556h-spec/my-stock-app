import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

# 1. 頁面配置
st.set_page_config(page_title="台股預測助手", layout="centered")

def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    name = f"股票 {sid}"
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search:
            name = title_search.group(1).split('-')[0].strip()
    except: pass
    return name

st.title("📈 台股精準預測 APP")
stock_id = st.text_input("輸入股票代碼 (如 8088):", value="8088")

if stock_id:
    ticker_str = f"{stock_id}.TWO" if int(stock_id) > 1000 else f"{stock_id}.TW"
    df = yf.download(ticker_str, period="60d", progress=False, auto_adjust=True)
    
    if not df.empty:
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close = df['Close'].ffill()
        high = df['High'].ffill()
        low = df['Low'].ffill()
        volume = df['Volume'].ffill()
        
        stock_name = get_clean_info(stock_id)
        
        # 2. 計算預測與準確率(漲跌幅)
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean()
        tp = (high + low + close) / 3
        mf_flow = np.where(tp > tp.shift(1), tp * volume, -tp * volume)
        
        today_close = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        
        # 預測邏輯
        pred_next = today_close + (atr_val * 0.8)
        pred_5day = today_close + (atr_val * 1.8)

        # 3. 頂部數據卡片 (含準確率/漲跌幅)
        st.subheader(f"📊 {stock_name} ({stock_id})")
        st.metric("今日收盤價", f"{today_close:.2f}")
        
        c1, c2 = st.columns(2)
        c1.metric("預估隔日最高", f"{pred_next:.2f}", f"{((pred_next/today_close)-1)*100:+.2f}%")
        c2.metric("預估五日最高", f"{pred_5day:.2f}", f"{((pred_5day/today_close)-1)*100:+.2f}%")

        # 4. 繪圖 (全面避開中文亂碼)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2.5, 1]})
        
        # 上圖：走勢與預測 (標籤用英文)
        ax1.plot(df.index, close, color='#1f77b4', linewidth=2, label="Close Price")
        ax1.scatter(df.index[-1], pred_next, color='orange', s=100, label="Next Day Forecast")
        ax1.scatter(df.index[-1], pred_5day, color='red', marker='*', s=200, label="5-Day Forecast")
        ax1.set_title(f"{stock_id} Price & Forecast", fontsize=16)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 下圖：資金流向 (原本不見的價量表)
        colors = ['#ff9999' if x > 0 else '#99ff99' for x in mf_flow]
        ax2.bar(df.index, mf_flow/1e8, color=colors)
        ax2.set_ylabel("Money Flow (100M)")
        ax2.grid(True, alpha=0.2)
        
        st.pyplot(fig)
        
        # 5. 用網頁文字補償圖中中文
        st.write("### 📔 圖表中文對照說明")
        st.write("- **藍線 (Close Price)**：每日收盤價走勢")
        st.write("- **橘點 (Next Day)**：預估隔日可能最高位")
        st.write("- **紅星 (5-Day)**：預估五日內可能最高位")
        st.write("- **下方紅綠柱**：資金流入/流出強度（紅漲綠跌）")
        
    else:
        st.error("查無資料")
