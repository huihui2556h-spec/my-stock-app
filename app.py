import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re

# 1. 頁面基礎設定
st.set_page_config(page_title="台股 AI 交易助手", layout="centered", page_icon="📈")

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

# --- 迎賓頁面與路由邏輯 ---
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

# 返回首頁按鈕
def go_home():
    st.session_state.mode = "home"
    st.rerun()

# --- 模式 A: 迎賓首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    st.image("https://cdn-icons-png.flaticon.com/512/2422/2422796.png", width=100)
    st.write("### 請選擇您今日的操作模式：")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時決策", use_container_width=True):
            st.session_state.mode = "realtime"
            st.rerun()
        st.caption("適合同步看盤。根據開盤價、即時量能給予秒級建議。")
        
    with col_b:
        if st.button("📊 波段數據預估", use_container_width=True):
            st.session_state.mode = "forecast"
            st.rerun()
        st.caption("適合盤後分析。計算隔日與五日達成率、壓力支撐位。")

# --- 模式 B: 盤中即時決策 (秒級/量能) ---
elif st.session_state.mode == "realtime":
    st.sidebar.button("⬅️ 返回首頁", on_click=go_home)
    st.title("⚡ 盤中即時量價決策")
    stock_id = st.text_input("輸入代碼 (例如: 8088):", placeholder="盤中建議對照即時看盤軟體...")
    
    if stock_id:
        with st.spinner('計算即時趨勢中...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            ticker = yf.Ticker(symbol)
            # 抓取 1分鐘線 (即時) 與 日線 (算波動)
            df_1m = ticker.history(interval="1m", period="1d")
            df_daily = ticker.history(period="20d")
            
            if not df_1m.empty and len(df_daily) > 1:
                curr_p = df_1m['Close'].iloc[-1]
                open_p = df_1m['Open'].iloc[0]
                prev_c = df_daily['Close'].iloc[-2]
                
                # 計算即時強弱
                st.subheader(f"📊 {get_clean_info(stock_id)} (即時監控)")
                m1, m2, m3 = st.columns(3)
                m1.metric("當前價", f"{curr_p:.2f}", f"{((curr_p/prev_c)-1)*100:+.2f}%")
                m2.metric("開盤價", f"{open_p:.2f}", f"跳空 {((open_p/prev_c)-1)*100:+.2f}%")
                vol_ratio = df_1m['Volume'].sum() / df_daily['Volume'].mean()
                m3.metric("相對量能", f"{vol_ratio:.2f}x")

                st.divider()
                # 實戰建議
                if curr_p > open_p and curr_p > prev_c:
                    st.success("🔥 **多頭強勢：建議守開盤價操作**")
                    st.write(f"👉 **建議買點**：**{open_p:.2f}** 附近")
                elif curr_p < prev_c:
                    st.error("❄️ **弱勢探底：不宜逆勢進場**")
                    st.write("👉 **警告**：目前股價在平盤以下且開低，翻紅機率低，建議觀望。")
                else:
                    st.info("⚖️ **區間震盪：等待量能突破**")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_1m.index, df_1m['Close'], color='#1f77b4', label="1-min Trend")
                ax.axhline(y=open_p, color='orange', linestyle='--', label="Open")
                ax.axhline(y=prev_c, color='gray', linestyle='--', label="Prev Close")
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("無法獲取即時數據。")

# --- 模式 C: 波段數據預估 (原本的邏輯) ---
elif st.session_state.mode == "forecast":
    st.sidebar.button("⬅️ 返回首頁", on_click=go_home)
    st.title("📊 波段數據預估")
    stock_id = st.text_input("輸入代碼 (例如: 2330):")

    if stock_id:
        with st.spinner('計算歷史達成率中...'):
            success = False
            for suffix in [".TW", ".TWO"]:
                df = yf.download(f"{stock_id}{suffix}", period="150d", progress=False, auto_adjust=True)
                if not df.empty and len(df) > 30:
                    success = True; break
            
            if success:
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                close, high, low = df['Close'].ffill(), df['High'].ffill(), df['Low'].ffill()
                tr = np.maximum(high-low, np.maximum(abs(high-close.shift(1)), abs(low-close.shift(1))))
                atr = tr.rolling(14).mean().fillna(method='bfill')
                
                # 回測
                acc_h1, acc_h5, acc_l1, acc_l5 = [], [], [], []
                for i in range(25, 5, -1):
                    p_c, p_a = close.iloc[-i], atr.iloc[-i]
                    act_h1, act_h5 = high.iloc[-i+1], high.iloc[-i+1:-i+6].max()
                    act_l1, act_l5 = low.iloc[-i+1], low.iloc[-i+1:-i+6].min()
                    if not np.isnan(act_h1):
                        acc_h1.append(min(act_h1/(p_c+(p_a*0.8)), 1.0))
                        acc_h5.append(min(act_h5/(p_c+(p_a*1.8)), 1.0))
                        acc_l1.append(min((p_c-(p_a*0.6))/act_l1, 1.0))
                        acc_l5.append(min((p_c-(p_a*1.5))/act_l5, 1.0))

                curr_c, curr_a = float(close.iloc[-1]), float(atr.iloc[-1])
                p_h1, p_h5 = curr_c + (curr_a * 0.8), curr_c + (curr_a * 1.8)
                p_l1, p_l5 = curr_c - (curr_a * 0.6), curr_c - (curr_a * 1.5)

                st.subheader(f"🏠 {get_clean_info(stock_id)}")
                # 壓力
                st.markdown("### 🎯 目標壓力位")
                c1, c2 = st.columns(2)
                c1.metric("📈 隔日最高", f"{p_h1:.2f}", f"漲幅 {((p_h1/curr_c)-1)*100:+.2f}%")
                c1.write(f"↳ 達成率：**{np.mean(acc_h1)*100:.1f}%**")
                c2.metric("🚩 五日最高", f"{p_h5:.2f}", f"漲幅 {((p_h5/curr_c)-1)*100:+.2f}%")
                c2.write(f"↳ 達成率：**{np.mean(acc_h5)*100:.1f}%**")

                # 支撐
                st.markdown("### 🛡️ 預估支撐位")
                c3, c4 = st.columns(2)
                c3.metric("📉 隔日最低", f"{p_l1:.2f}", f"跌幅 {((p_l1/curr_c)-1)*100:+.2f}%", delta_color="inverse")
                c3.write(f"↳ 達成率：**{np.mean(acc_l1)*100:.1f}%**")
                c4.metric("⚓ 五日最低", f"{p_l5:.2f}", f"跌幅 {((p_l5/curr_c)-1)*100:+.2f}%", delta_color="inverse")
                c4.write(f"↳ 達成率：**{np.mean(acc_l5)*100:.1f}%**")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index[-40:], close.tail(40), label="Price", color='#1f77b4')
                ax.axhline(y=p_h5, color='red', linestyle='--', alpha=0.3)
                ax.axhline(y=p_l5, color='green', linestyle='--', alpha=0.3)
                st.pyplot(fig)
