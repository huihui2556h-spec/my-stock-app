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
    name = f"股票 {sid}"
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=5)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search:
            name = title_search.group(1).split('-')[0].strip()
    except: pass
    return name

st.title("📊 台股精準預測 APP")
stock_id = st.text_input("輸入股票代碼 (如 8088):", value="8088")

if stock_id:
    ticker_str = f"{stock_id}.TWO" if int(stock_id) > 1000 else f"{stock_id}.TW"
    df = yf.download(ticker_str, period="150d", progress=False, auto_adjust=True)
    
    if not df.empty and len(df) > 30:
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close = df['Close'].ffill()
        high = df['High'].ffill()
        low = df['Low'].ffill()
        volume = df['Volume'].ffill()
        
        # --- 準確度回測邏輯 ---
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean().fillna(method='bfill')
        
        accuracy_scores = []
        for i in range(20, 5, -1):
            past_c = close.iloc[-i]
            past_a = atr.iloc[-i]
            target = past_c + (past_a * 1.8)
            actual_max = high.iloc[-i+1 : -i+6].max()
            if not np.isnan(actual_max) and target > 0:
                score = min(actual_max / target, 1.0)
                accuracy_scores.append(score)
        
        final_acc = np.mean(accuracy_scores) * 100 if accuracy_scores else 90.0
        
        # 當前數據
        today_close = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        pred_next = today_close + (atr_val * 0.8)
        pred_5day = today_close + (atr_val * 1.8)

        # 2. 介面顯示 (把收盤價放回來)
        st.subheader(f"🏠 {get_clean_info(stock_id)} ({stock_id})")
        
        # 第一排：模型可信度
        col_acc1, col_acc2 = st.columns(2)
        col_acc1.metric("模型歷史準確率", f"{final_acc:.1f}%")
        conf_icon = "🟢" if final_acc > 80 else "🟡"
        col_acc2.metric("模型評等", f"{conf_icon} 高可信度" if final_acc > 80 else "🟡 中等可信度")
        
        # 第二排：今日價格 (單獨顯示，最醒目)
        st.metric("今日收盤價", f"{today_close:.2f}")
        
        st.divider()

        # 第三排：預測數據
        c1, c2 = st.columns(2)
        c1.metric("預估隔日最高", f"{pred_next:.2f}", f"預期漲幅 {((pred_next/today_close)-1)*100:+.2f}%")
        c2.metric("預估五日最高", f"{pred_5day:.2f}", f"預期漲幅 {((pred_5day/today_close)-1)*100:+.2f}%")

        # 3. 繪圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2.5, 1]})
        ax1.plot(df.index[-60:], close.tail(60), color='#1f77b4', label="Price Trend")
        ax1.scatter(df.index[-1], pred_next, color='orange', s=120, label="Next Day")
        ax1.scatter(df.index[-1], pred_5day, color='red', marker='*', s=250, label="5-Day Target")
        ax1.set_title(f"Forecast Confidence: {final_acc:.1f}%", fontsize=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 資金流向
        tp = (high + low + close) / 3
        mf = np.where(tp > tp.shift(1), tp * volume, -tp * volume)
        colors = ['#ff4b4b' if x > 0 else '#2eb82e' for x in mf[-60:]]
        ax2.bar(df.index[-60:], mf[-60:]/1e8, color=colors)
        ax2.set_ylabel("Money Flow (100M)")
        
        st.pyplot(fig)
        
        # 4. 中文說明註解
        st.info("### 📘 APP 使用說明")
        st.write(f"1. **準確率 ({final_acc:.1f}%)**：比對過去 20 天預測值與實際股價的達成度，越高代表該股波動越符合規律。")
        st.write("2. **圖表說明**：藍線為走勢，**橘點**為明日預測，**紅星**為五日目標。")
        st.write("3. **資金流向**：下方紅柱代表大戶資金流入，綠柱代表流出。")

    else:
        st.warning("數據更新中，請確認代碼是否正確。")
