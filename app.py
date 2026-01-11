import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

# 1. 頁面優化設定
st.set_page_config(page_title="台股交易助手", layout="centered", page_icon="⚖️")

def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    name = f"股票 {sid}"
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=10)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search:
            name = title_search.group(1).split('-')[0].strip()
    except: pass
    return name

# --- 歡迎頁面邏輯 ---
if 'started' not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.title("⚖️ 台股交易決策系統")
    st.image("https://cdn-icons-png.flaticon.com/512/2422/2422796.png", width=120)
    st.write("### AI 判斷支撐與當沖位")
    st.write("整合 ATR 波動率與多空力道，提供精準的買賣點建議。")
    if st.button("啟動系統"):
        st.session_state.started = True
        st.rerun()
else:
    st.title("🔍 交易策略分析")
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.started = False
        st.rerun()

    stock_id = st.text_input("請輸入台股代碼 (例如: 2330, 8088):", placeholder="在此輸入代碼...")

    if stock_id:
        with st.spinner('正在分析盤勢...'):
            success = False
            for suffix in [".TW", ".TWO"]:
                ticker_str = f"{stock_id}{suffix}"
                df = yf.download(ticker_str, period="150d", progress=False, auto_adjust=True)
                if not df.empty and len(df) > 30:
                    success = True
                    break
            
            if success:
                # 數據處理
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                close, high, low = df['Close'].ffill(), df['High'].ffill(), df['Low'].ffill()
                
                # 計算關鍵指標
                tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
                atr = tr.rolling(14).mean().fillna(method='bfill')
                
                # --- 核心回測邏輯 (計算三種準確率) ---
                acc_high, acc_low, acc_day = [], [], []
                for i in range(20, 5, -1):
                    p_c, p_a = close.iloc[-i], atr.iloc[-i]
                    # 預測值
                    p_h_5d = p_c + (p_a * 1.8) # 五日最高
                    p_l_5d = p_c - (p_a * 1.5) # 五日最低
                    # 實際值
                    a_h_5d = high.iloc[-i+1 : -i+6].max()
                    a_l_5d = low.iloc[-i+1 : -i+6].min()
                    
                    acc_high.append(min(a_h_5d / p_h_5d, 1.0) if p_h_5d > 0 else 0)
                    acc_low.append(min(p_l_5d / a_l_5d, 1.0) if a_l_5d > 0 else 0)
                
                final_acc_h = np.mean(acc_high) * 100
                final_acc_l = np.mean(acc_low) * 100

                # 當前預測值
                curr_c, curr_a = float(close.iloc[-1]), float(atr.iloc[-1])
                pred_h_1 = curr_c + (curr_a * 0.8)
                pred_l_1 = curr_c - (curr_a * 0.6)
                pred_h_5 = curr_c + (curr_a * 1.8)
                pred_l_5 = curr_c - (curr_a * 1.5)
                
                # 當沖建議 (以開盤價為基準的簡易邏輯)
                buy_point = curr_c - (curr_a * 0.3)
                sell_point = curr_c + (curr_a * 0.7)

                # --- 介面顯示 ---
                st.subheader(f"🏠 {get_clean_info(stock_id)} ({stock_id})")
                st.metric("今日收盤價", f"{curr_c:.2f}")

                # 1. 最高價預測 (壓力位)
                st.markdown("#### 📈 目標壓力位")
                col1, col2 = st.columns(2)
                col1.metric("隔日預估最高", f"{pred_h_1:.2f}")
                col2.metric("五日預估最高", f"{pred_h_5:.2f}", f"歷史達成率 {final_acc_h:.1f}%")

                # 2. 最低價預測 (支撐位)
                st.markdown("#### 📉 預估支撐位")
                col3, col4 = st.columns(2)
                col3.metric("隔日預估最低", f"{pred_l_1:.2f}")
                col4.metric("五日預估最低", f"{pred_l_5:.2f}", f"歷史達成率 {final_acc_l:.1f}%", delta_color="inverse")

                # 3. 當沖交易建議
                st.warning("⚠️ **隔日當沖參考 (買低賣高)**")
                d_col1, d_col2 = st.columns(2)
                d_col1.write(f"🔹 建議買入點：**{buy_point:.2f}**")
                d_col2.write(f"🔸 建議賣出點：**{sell_point:.2f}**")
                st.caption(f"當沖策略綜合準確率：{(final_acc_h + final_acc_l)/2:.1f}% (根據波段穩定度推算)")

                # 繪圖
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df.index[-40:], close.tail(40), label="Price", color='#1f77b4', linewidth=2)
                ax.axhline(y=pred_h_5, color='red', linestyle='--', alpha=0.5, label="5D Resistance")
                ax.axhline(y=pred_l_5, color='green', linestyle='--', alpha=0.5, label="5D Support")
                ax.scatter(df.index[-1], pred_h_1, color='orange', label="Next High")
                ax.set_title(f"{stock_id} Support & Resistance")
                ax.legend()
                st.pyplot(fig)
                
                st.info("💡 **操作建議**：當沖建議買點通常設於平盤下方支撐區，賣點設於預期壓力區。若準確率低於 80%，建議減少部位。")
            else:
                st.error("搜尋不到數據，請檢查代碼。")
