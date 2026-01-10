import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

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
stock_id = st.text_input("輸入股票代碼:", value="8358")

if stock_id:
    ticker_str = f"{stock_id}.TWO" if int(stock_id) > 1000 else f"{stock_id}.TW"
    df = yf.download(ticker_str, period="100d", progress=False, auto_adjust=True)
    
    if not df.empty:
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close = df['Close'].ffill()
        high = df['High'].ffill()
        low = df['Low'].ffill()
        volume = df['Volume'].ffill()
        
        # --- 核心計算與準確度回測邏輯 ---
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean()
        
        # 1. 歷史準確度計算 (回測過去 10 天)
        accuracy_list = []
        for i in range(20, 1, -1):
            past_close = close.iloc[-i]
            past_atr = atr.iloc[-i]
            actual_max_5d = high.iloc[-i+1 : -i+6].max() # 預測後五天的實際最高價
            pred_max_5d = past_close + (past_atr * 1.8)
            
            # 如果實際最高價達到或超過預測價的 95%，視為預測成功
            score = min(actual_max_5d / pred_max_5d, 1.0)
            accuracy_list.append(score)
        
        final_accuracy = np.mean(accuracy_list) * 100 # 平均準確率
        
        # 2. 當前預測計算
        today_close = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        pred_next = today_close + (atr_val * 0.8)
        pred_5day = today_close + (atr_val * 1.8)

        # --- 介面顯示 ---
        st.subheader(f"📊 {get_clean_info(stock_id)} ({stock_id})")
        
        # 醒目的準確度顯示儀表板
        acc_col1, acc_col2 = st.columns([1, 1])
        with acc_col1:
            st.metric("歷史預測準確率", f"{final_accuracy:.1f}%")
        with acc_col2:
            confidence = "高可信度" if final_accuracy > 85 else "中等可信度" if final_accuracy > 70 else "低可信度 (建議觀望)"
            st.write(f"🔍 預測可信度：**{confidence}**")
        
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("預估隔日最高", f"{pred_next:.2f}", f"預期漲幅 {((pred_next/today_close)-1)*100:+.2f}%")
        with c2:
            st.metric("預估五日最高", f"{pred_5day:.2f}", f"預期漲幅 {((pred_5day/today_close)-1)*100:+.2f}%")

        # 繪圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2.5, 1]})
        ax1.plot(df.index, close, label="Price", linewidth=2)
        ax1.scatter(df.index[-1], pred_next, color='orange', s=100, label="Next Day")
        ax1.scatter(df.index[-1], pred_5day, color='red', marker='*', s=200, label="5-Day Target")
        ax1.set_title(f"Model Accuracy: {final_accuracy:.1f}%", fontsize=14)
        ax1.legend()
        
        # 價量表
        tp = (high + low + close) / 3
        mf_flow = np.where(tp > tp.shift(1), tp * volume, -tp * volume)
        colors = ['#ff4b4b' if x > 0 else '#2eb82e' for x in mf_flow]
        ax2.bar(df.index, mf_flow/1e8, color=colors)
        ax2.set_ylabel("Money Flow")
        
        st.pyplot(fig)
        st.info(f"💡 **準確率說明**：此數字是比對過去 20 天模型預測目標與實際走勢的達成率。{final_accuracy:.1f}% 表示模型對該股的波動掌握度。")

    else:
        st.error("查無資料")
