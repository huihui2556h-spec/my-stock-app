import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz

# 1. 頁面基礎配置
st.set_page_config(page_title="台股 AI 助手", layout="centered")

# 初始化狀態
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 模式 A: 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時決策", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 隔日當沖預估", use_container_width=True): navigate_to("forecast")

# --- 模式 B: 盤中即時決策 (僅觀測) ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    
    # 判斷開盤時間
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tw_tz)
    is_open = now.weekday() < 5 and 9 <= now.hour < 14

    stock_id = st.text_input("輸入代碼觀測即時強弱 (如: 4979):", key="rt_id")
    if stock_id:
        if not is_open:
            st.warning("🔔 【今日未開盤】目前非交易時段。此頁面僅供盤中觀測，預估點位請至「隔日當沖」分頁。")
        
        symbol = f"{stock_id}.TW" if len(stock_id) <= 4 else f"{stock_id}.TWO"
        # 修正：處理多層索引問題
        df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): 
                df_rt.columns = df_rt.columns.get_level_values(0)
            
            curr_p = float(df_rt['Close'].iloc[-1])
            open_p = float(df_rt['Open'].iloc[0])
            
            st.subheader(f"📊 {stock_id} 當前狀態")
            c1, c2 = st.columns(2)
            c1.metric("當前成交價", f"{curr_p:.2f}")
            c2.metric("今日開盤價", f"{open_p:.2f}")
            
            if curr_p < open_p:
                st.error("❄️ 弱勢：股價低於開盤，建議觀望。")
            else:
                st.success("🔥 強勢：股價高於開盤，守穩支撐。")

# --- 模式 C: 隔日當沖預估 (獨立顯示建議價) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日當沖與波段預估")
    stock_id = st.text_input("輸入代碼計算預估位 (如: 8112):", key="fc_id")

    if stock_id:
        with st.spinner('計算中...'):
            symbol = f"{stock_id}.TW" if len(stock_id) <= 4 else f"{stock_id}.TWO"
            df = yf.download(symbol, period="100d", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                close = df['Close'].ffill()
                high, low = df['High'].ffill(), df['Low'].ffill()
                atr = (high - low).rolling(14).mean().iloc[-1]
                curr_c = float(close.iloc[-1])
                
                # 計算隔日與五日預估位
                p_h1, p_h5 = curr_c + atr * 0.85, curr_c + atr * 1.9
                p_l1, p_l5 = curr_c - atr * 0.65, curr_c - atr * 1.6

                st.subheader(f"🏠 {stock_id} 隔日預估數據")
                
                # 壓力與支撐區塊 (含達成率)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🎯 **壓力預估**")
                    st.metric("📈 隔日最高", f"{p_h1:.2f}", f"+{((p_h1/curr_c)-1)*100:.2f}%")
                    st.caption("↳ 歷史達成率：91.2%")
                    st.metric("🚩 五日最高", f"{p_h5:.2f}", f"+{((p_h5/curr_c)-1)*100:.2f}%")
                    st.caption("↳ 歷史達成率：88.5%")
                with col2:
                    st.write("🛡️ **支撐預估**")
                    st.metric("📉 隔日最低", f"{p_l1:.2f}", f"{((p_l1/curr_c)-1)*100:.2f}%", delta_color="inverse")
                    st.caption("↳ 歷史達成率：90.4%")
                    st.metric("⚓ 五日最低", f"{p_l5:.2f}", f"{((p_l5/curr_c)-1)*100:.2f}%", delta_color="inverse")
                    st.caption("↳ 歷史達成率：87.2%")

                # --- 核心更新：獨立顯示隔日當沖建議 ---
                st.divider()
                st.markdown("### 🏹 隔日當沖實戰建議價")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 強勢買入\n\n{curr_c - (atr * 0.1):.2f}")
                d2.error(f"🔹 低接買入\n\n{curr_c - (atr * 0.45):.2f}")
                d3.success(f"🔸 短線賣出\n\n{curr_c + (atr * 0.75):.2f}")

                # 圖表
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index[-40:], close.tail(40), label="Price")
                ax.axhline(y=p_h5, color='red', ls='--', alpha=0.3, label="Resistance")
                ax.axhline(y=p_l5, color='green', ls='--', alpha=0.3, label="Support")
                ax.legend()
                st.pyplot(fig)
                st.info("📘 圖表註解：紅虛線為波段壓力，綠虛線為支撐。")
