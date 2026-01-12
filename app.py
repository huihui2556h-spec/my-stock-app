import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta
import pytz

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🧬 外部籌碼資料庫：安全載入模組 ---
def get_institutional_chips(stock_id):
    """抓取三大法人與融資融券，計算籌碼修正因子 (含防錯機制)"""
    try:
        # 在函式內部載入，避免一啟動就崩潰
        from FinMind.data import DataLoader
        dl = DataLoader()
        start_dt = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        
        # 獲取法人資料
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        # 獲取融資融券
        margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_dt)
        
        chip_weight = 1.0 
        
        if not inst_df.empty:
            recent = inst_df.tail(9) 
            net = recent['buy'].sum() - recent['sell'].sum()
            # 法人買超則增加信心權重 (考慮波動慣性)
            if net > 0: chip_weight += 0.008
            else: chip_weight -= 0.008
            
        if not margin_df.empty:
            m_data = margin_df.tail(3)
            # 融資減少視為籌碼集中，利於多頭
            if m_data['Margin_Purchase_today_balance'].iloc[-1] < m_data['Margin_Purchase_today_balance'].iloc[0]:
                chip_weight += 0.003
        
        return chip_weight
    except Exception as e:
        # 如果 FinMind 還沒裝好或 API 失敗，返回 1.0 確保原本價量預測依然可用
        return 1.0

# --- 🧠 AI 動態特徵預測核心 (結合 "波動慣性" 與 "法人籌碼") ---
def ai_dynamic_forecast(df, chip_f=1.0):
    try:
        # 學習該股近 60 日歷史波動特徵
        df_clean = df.tail(60).copy()
        df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        
        # 使用百分位數計算動態區間，並納入法人籌碼修正
        h1_p = df_clean['h_pct'].quantile(0.75) * chip_f
        h5_p = df_clean['h_pct'].quantile(0.95) * chip_f
        l1_p = df_clean['l_pct'].quantile(0.25) / chip_f
        l5_p = df_clean['l_pct'].quantile(0.05) / chip_f
        
        return h1_p, h5_p, l1_p, l5_p
    except:
        return 0.02, 0.05, -0.015, -0.04

def calculate_real_accuracy(df, target_p, side='high'):
    try:
        df_c = df.copy().tail(60)
        hits = 0
        for i in range(1, len(df_c)):
            prev_c = df_c['Close'].iloc[i-1]
            actual = df_c['High'].iloc[i] if side == 'high' else df_c['Low'].iloc[i]
            pred = prev_c * (1 + target_p)
            if side == 'high' and actual >= pred: hits += 1
            elif side == 'low' and actual <= pred: hits += 1
        return (hits / len(df_c)) * 100
    except: return 0.0

def get_stock_name(sid):
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"股票 {sid}"

def render_box(label, price, pct, acc, color="red"):
    c_code = "#FF4B4B" if color == "red" else "#28A745"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {c_code}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{c_code}; color:white; padding:2px 8px; border-radius:5px; font-size:13px;">
                預估振幅：{pct:.2f}%
            </span>
            <p style="margin-top:10px; font-size:11px; color:#888;">↳ 歷史特徵達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 🚀 頁面路由 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 隔日深度預估", use_container_width=True): navigate_to("forecast")

# =========================================================
# ⚡ 盤中即時 (保留重整按鈕與 VWAP)
# =========================================================
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    col_h, col_r = st.columns([4, 1.2])
    col_h.title("⚡ 盤中動態決策")
    if col_r.button("🔄 點擊重整", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    sid_rt = st.text_input("輸入台股代碼:", key="rt_id_fixed")
    if sid_rt:
        success = False
        for suf in [".TW", ".TWO"]:
            df_rt = yf.download(f"{sid_rt}{suf}", period="1d", interval="1m", progress=False)
            if not df_rt.empty:
                success = True; break
        if success:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            cp = float(df_rt['Close'].iloc[-1])
            vp = float(df_rt['VWAP'].iloc[-1])
            st.subheader(f"🎯 {get_stock_name(sid_rt)}")
            st.metric("即時現價", f"{cp:.2f}")
            st.success(f"🔹 即時支撐 (VWAP)：{vp:.2f}")
            st.error(f"🔸 即時建議停利：{cp * 1.015:.2f}")
        else: st.warning("目前無即時成交數據。")

# =========================================================
# 📊 隔日深度預估 (整合 波動慣性 與 法人籌碼)
# =========================================================
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日多因子 AI 預判")
    sid_fc = st.text_input("輸入台股代碼:", key="fc_id_fixed")
    if sid_fc:
        with st.spinner('AI 正在計算波動慣性並整合法人籌碼數據...'):
            success = False
            for suf in [".TW", ".TWO"]:
                df = yf.download(f"{sid_fc}{suf}", period="100d", progress=False)
                if not df.empty:
                    success = True; break
            
            if success:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                curr_close = float(df['Close'].iloc[-1])
                
                # 1. 獲取籌碼權重
                c_weight = get_institutional_chips(sid_fc)
                
                # 2. AI 預測點位
                h1, h5, l1, l5 = ai_dynamic_forecast(df, chip_f=c_weight)
                
                ph1, ph5 = curr_close*(1+h1), curr_close*(1+h5)
                pl1, pl5 = curr_close*(1+l1), curr_close*(1+l5)

                st.subheader(f"🏠 {get_stock_name(sid_fc)}")
                st.metric("最新收盤價", f"{curr_close:.2f}")
                
                # 顯示分析狀態
                chip_desc = "🔥 籌碼面優良 (法人買進)" if c_weight > 1 else "❄️ 籌碼面平淡"
                st.info(f"**AI 綜合診斷：{chip_desc} (權重: {c_weight:.3f})**")
                
                st.divider()
                cola, colb = st.columns(2)
                with cola:
                    render_box("📈 隔日最高預估", ph1, h1*100, calculate_real_accuracy(df, h1, 'high'), "red")
                    render_box("🚩 五日最高預估", ph5, h5*100, calculate_real_accuracy(df, h5, 'high'), "red")
                with colb:
                    render_box("📉 隔日最低預估", pl1, l1*100, calculate_real_accuracy(df, l1, 'low'), "green")
                    render_box("⚓ 五日最低預估", pl5, l5*100, calculate_real_accuracy(df, l5, 'low'), "green")

                # 圖表
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
                ax1.plot(df.index[-40:], df['Close'].tail(40), color='#1f77b4', lw=2)
                ax1.axhline(y=ph1, color='red', ls='--', alpha=0.4, label="AI Resistance")
                ax1.axhline(y=pl1, color='green', ls='--', alpha=0.4, label="AI Support")
                ax1.legend()
                
                pdf = df.tail(40)
                clrs = ['red' if pdf['Close'].iloc[i] >= pdf['Open'].iloc[i] else 'green' for i in range(len(pdf))]
                ax2.bar(pdf.index, pdf['Volume'], color=clrs, alpha=0.6)
                st.pyplot(fig)
                
                st.info("💡 **AI 邏輯更新**：此預測結果已同步考慮「歷史波動慣性」與「法人籌碼集中度」。")
