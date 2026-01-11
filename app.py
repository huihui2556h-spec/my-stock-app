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

# --- 迎賓頁面邏輯 ---
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def go_home():
    st.session_state.mode = "home"
    st.rerun()

# --- 模式 A: 迎賓首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易系統")
    st.write("### 請選擇今日操作模式：")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時決策", use_container_width=True):
            st.session_state.mode = "realtime"
            st.rerun()
    with col_b:
        if st.button("📊 波段數據預估", use_container_width=True):
            st.session_state.mode = "forecast"
            st.rerun()

# --- 模式 B: 盤中即時決策 ---
elif st.session_state.mode == "realtime":
    st.sidebar.button("⬅️ 返回首頁", on_click=go_home)
    st.title("⚡ 盤中即時量價")
    stock_id = st.text_input("輸入代碼:", key="rt_id")
    if stock_id:
        symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
        df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
        df_hist = yf.download(symbol, period="5d", progress=False)
        if not df_rt.empty:
            curr_p = df_rt['Close'].iloc[-1]
            open_p = df_rt['Open'].iloc[0]
            prev_c = df_hist['Close'].iloc[-2]
            st.subheader(f"📊 {get_clean_info(stock_id)}")
            m1, m2 = st.columns(2)
            m1.metric("當前價", f"{curr_p:.2f}", f"{((curr_p/prev_c)-1)*100:+.2f}%")
            m2.metric("開盤價", f"{open_p:.2f}", f"跳空 {((open_p/prev_c)-1)*100:+.2f}%")
            if curr_p > open_p: st.success("🔥 強勢：守開盤價操作")
            else: st.error("❄️ 弱勢：破平盤觀望")

# --- 模式 C: 波段數據預估 (修復達成率與圖表) ---
elif st.session_state.mode == "forecast":
    st.sidebar.button("⬅️ 返回首頁", on_click=go_home)
    st.title("📊 波段數據預估")
    stock_id = st.text_input("輸入代碼:", key="fc_id")
    if stock_id:
        with st.spinner('數據計算中...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            df = yf.download(symbol, period="100d", progress=False)
            if not df.empty:
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                close, high, low = df['Close'].ffill(), df['High'].ffill(), df['Low'].ffill()
                # ATR
                tr = np.maximum(high-low, np.maximum(abs(high-close.shift(1)), abs(low-close.shift(1))))
                atr = tr.rolling(14).mean().fillna(method='bfill')
                
                # 達成率回測 (加入保護防止亂碼)
                acc_list = {"h1":[], "h5":[], "l1":[], "l5":[]}
                for i in range(20, 5, -1):
                    p_c, p_a = close.iloc[-i], atr.iloc[-i]
                    # 避免分母為0或NaN
                    if p_a > 0:
                        acc_list["h1"].append(min(high.iloc[-i+1] / (p_c + p_a*0.8), 1.0))
                        acc_list["h5"].append(min(high.iloc[-i+1:-i+6].max() / (p_c + p_a*1.8), 1.0))
                        acc_list["l1"].append(min((p_c - p_a*0.6) / low.iloc[-i+1], 1.0))
                        acc_list["l5"].append(min((p_c - p_a*1.5) / low.iloc[-i+1:-i+6].min(), 1.0))
                
                def get_acc(key): 
                    val = np.mean([x for x in acc_list[key] if not np.isnan(x)]) * 100
                    return val if not np.isnan(val) else 0.0

                curr_c, curr_a = float(close.iloc[-1]), float(atr.iloc[-1])
                # 介面顯示
                st.subheader(f"🏠 {get_clean_info(stock_id)}")
                st.write(f"今日收盤價：**{curr_c:.2f}**")
                
                # 壓力位
                st.markdown("### 🎯 目標壓力位")
                c1, c2 = st.columns(2)
                p_h1 = curr_c + curr_a*0.8
                c1.metric("📈 隔日最高", f"{p_h1:.2f}", f"漲幅 {((p_h1/curr_c)-1)*100:+.2f}%")
                c1.write(f"↳ 達成率：**{get_acc('h1'):.1f}%**")
                p_h5 = curr_c + curr_a*1.8
                c2.metric("🚩 五日最高", f"{p_h5:.2f}", f"漲幅 {((p_h5/curr_c)-1)*100:+.2f}%")
                c2.write(f"↳ 達成率：**{get_acc('h5'):.1f}%**")

                # 支撐位
                st.markdown("### 🛡️ 預估支撐位")
                c3, c4 = st.columns(2)
                p_l1 = curr_c - curr_a*0.6
                c3.metric("📉 隔日最低", f"{p_l1:.2f}", f"跌幅 {((p_l1/curr_c)-1)*100:+.2f}%", delta_color="inverse")
                c3.write(f"↳ 達成率：**{get_acc('l1'):.1f}%**")
                p_l5 = curr_c - curr_a*1.5
                c4.metric("⚓ 五日最低", f"{p_l5:.2f}", f"跌幅 {((p_l5/curr_c)-1)*100:+.2f}%", delta_color="inverse")
                c4.write(f"↳ 達成率：**{get_acc('l5'):.1f}%**")

                # 圖表展示
                st.divider()
                st.write("### 📉 走勢與法人籌碼")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax1.plot(df.index[-40:], close.tail(40), label="Price", color='#1f77b4', linewidth=2)
                ax1.axhline(y=p_h5, color='red', linestyle='--', alpha=0.3, label="Resistance")
                ax1.axhline(y=p_l5, color='green', linestyle='--', alpha=0.3, label="Support")
                ax1.legend()
                
                # 模擬法人買賣超圖表 (yfinance 無直接籌碼，用成交量變化模擬趨勢)
                v_change = df['Volume'].diff().tail(40)
                colors = ['red' if x > 0 else 'green' for x in v_change]
                ax2.bar(df.index[-40:], v_change, color=colors, alpha=0.7, label="Volume Change")
                ax2.set_title("Volume Momentum (Proxy for Net Buy/Sell)")
                st.pyplot(fig)

                st.divider()
                st.subheader("📘 數據與註解說明")
                st.markdown(f"""
                **1. 圖表標籤對照：**
                * **Price (藍線)**：歷史收盤價。
                * **Resistance (紅虛線)**：五日預期最高壓力。
                * **Support (綠虛線)**：五日預期最低支撐。
                * **Volume Momentum (柱狀圖)**：成交動能（紅買超/綠賣超傾向）。

                **2. 達成率說明：**
                * 達成率代表模型過去預測目標價與實際最高/最低價的吻合程度。

                **3. 實戰建議：**
                * **開盤強弱**：若開盤 > {curr_c:.2f} 且量大，優先看目標壓力；若開盤破平盤，則應觀察支撐。
                """)
