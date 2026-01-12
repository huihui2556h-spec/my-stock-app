import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
import requests
import re

# =========================================================
# 1. 系統環境設定
# =========================================================
st.set_page_config(page_title="台股 AI 多因子交易系統", layout="centered")

# 初始化頁面導航狀態
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    """【導航函數】處理頁面切換邏輯"""
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 核心運算引擎 (誤差補償 + 籌碼因子)
# =========================================================

def get_error_bias(df, days=10):
    """【誤差補償】計算過去10天預估偏離率，用來動態修正今日點位"""
    try:
        temp = df.copy().tail(days + 15)
        temp['ATR'] = (temp['High'] - temp['Low']).rolling(14).mean()
        biases = []
        for i in range(1, days + 1):
            prev_c = temp['Close'].iloc[-i-1]
            prev_atr = temp['ATR'].iloc[-i-1]
            actual_h = temp['High'].iloc[-i]
            if prev_atr > 0:
                biases.append(actual_h / (prev_c + prev_atr * 0.85))
        return np.mean(biases) if biases else 1.0
    except: return 1.0

def get_chip_factor(stock_id):
    """【FinMind 籌碼】獲取法人近5日買賣超慣性 (2026-01-12 指令)"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            return (1.025, "✅ 籌碼面：法人偏多 (有利多頭慣性)") if net_buy > 0 else (0.975, "⚠️ 籌碼面：法人偏空 (注意回檔壓力)")
    except: pass
    return 1.0, "ℹ️ 籌碼面：中性 (暫無異常慣性)"

def calculate_real_accuracy(df, atr_factor, side='high'):
    """【AI 回測】計算過去 60 天點位的歷史命中達成率"""
    try:
        temp = df.copy().ffill()
        if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.get_level_values(0)
        backtest_days = min(len(temp) - 15, 60)
        hits = 0
        temp['ATR_CALC'] = (temp['High'] - temp['Low']).rolling(14).mean()
        for i in range(1, backtest_days + 1):
            idx = -i
            p_c, p_a = temp['Close'].iloc[idx-1], temp['ATR_CALC'].iloc[idx-1]
            actual = temp['High'].iloc[idx] if side == 'high' else temp['Low'].iloc[idx]
            pred = p_c + (p_a * atr_factor) if side == 'high' else p_c - (p_a * atr_factor)
            if (side == 'high' and actual >= pred) or (side == 'low' and actual <= pred): hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

def get_stock_name(stock_id):
    """抓取 Yahoo 財經股票中文名稱"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# =========================================================
# 3. 頁面介面邏輯
# =========================================================

# --- 🏠 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統 Pro")
    st.write("已整合：盤中監控、FinMind 籌碼因子、1日/5日波段全景分析")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

# --- ⚡ 盤中即時頁面 ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時監控")
    
    sid_rt = st.text_input("輸入股票代碼 (例: 2330):", key="rt_id")
    if sid_rt:
        with st.spinner('抓取即時數據中...'):
            df_rt = yf.download(f"{sid_rt}.TW", period="5d", interval="1m", progress=False)
            if df_rt.empty: df_rt = yf.download(f"{sid_rt}.TWO", period="5d", interval="1m", progress=False)
            
            if not df_rt.empty:
                if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
                st.subheader(f"🏠 {get_stock_name(sid_rt)} ({sid_rt})")
                
                curr_p = df_rt['Close'].iloc[-1]
                open_p = df_rt['Open'].iloc[0]
                
                c1, c2 = st.columns(2)
                c1.metric("當前價格", f"{curr_p:.2f}", delta=f"{curr_p - open_p:.2f}")
                c2.metric("今日估量", f"{int(df_rt['Volume'].sum()):,}")
                
                st.line_chart(df_rt['Close'].tail(100))
            else:
                st.error("找不到該標的數據。")

# --- 📊 深度預估頁面 (已取消分頁，改為垂直全覽) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    sid_fc = st.text_input("輸入分析代碼 (例: 8088):", key="fc_id")

    if sid_fc:
        with st.spinner('AI 進行因子整合與誤差補償中...'):
            df = None
            for suf in [".TW", ".TWO"]:
                tmp = yf.download(f"{sid_fc}{suf}", period="200d", progress=False)
                if not tmp.empty: df = tmp; break
            
            if df is not None:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # 計算運算因子
                chip_f, chip_msg = get_chip_factor(sid_fc) # 籌碼
                err_f = get_error_bias(df)                 # 誤差補償
                total_f = chip_f * err_f
                curr_c = float(df['Close'].iloc[-1])
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                
                # 點位計算
                ph1, pl1 = curr_c + (atr * 0.85 * total_f), curr_c - (atr * 0.65 / total_f)
                ph5, pl5 = curr_c + (atr * 1.90 * total_f), curr_c - (atr * 1.60 / total_f)

                st.subheader(f"🏠 {get_stock_name(sid_fc)} ({sid_fc})")
                st.info(f"{chip_msg} | AI 誤差修正: {err_f:.3f}")

                # --- 🎯 隔日預估區塊 ---
                st.markdown("### 🎯 隔日預估 (Short-term)")
                c1, c2 = st.columns(2)
                acc_h1 = calculate_real_accuracy(df, 0.85, 'high')
                acc_l1 = calculate_real_accuracy(df, 0.65, 'low')
                c1.error(f"**📈 隔日壓力** \n## {ph1:.2f} \n<small>AI 達成率: {acc_h1:.1f}%</small>")
                c2.success(f"**📉 隔日支撐** \n## {pl1:.2f} \n<small>AI 達成率: {acc_l1:.1f}%</small>")
                
                st.divider()

                # --- 🚩 五日預估區塊 ---
                st.markdown("### 🚩 五日波段 (Swing)")
                c3, c4 = st.columns(2)
                acc_h5 = calculate_real_accuracy(df, 1.90, 'high')
                acc_l5 = calculate_real_accuracy(df, 1.60, 'low')
                c3.error(f"**📈 五日最大壓力** \n## {ph5:.2f} \n<small>AI 達成率: {acc_h5:.1f}%</small>")
                c4.success(f"**📉 五日最大支撐** \n## {pl5:.2f} \n<small>AI 達成率: {acc_l5:.1f}%</small>")

                # --- 🏹 明日當沖指引 ---
                st.divider()
                st.markdown("### 🏹 明日當沖建議點位")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 **追多買點**: {curr_c + (atr * 0.15):.2f}")
                d2.warning(f"🔹 **低階支撐**: {curr_c - (atr * 0.45):.2f}")
                d3.success(f"🔸 **短線目標**: {curr_c + (atr * 0.75):.2f}")
                
                # --- 📊 趨勢圖表 ---
                st.divider()
                st.write("📊 價量趨勢與 AI 波段參考圖 (Price & Volume Action)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.tail(40).index, df.tail(40)['Close'], label="Close Price", color='#1f77b4', lw=2)
                ax.axhline(y=ph5, color='red', ls='--', alpha=0.5, label="5D Resistance")
                ax.axhline(y=pl5, color='green', ls='--', alpha=0.5, label="5D Support")
                ax.legend(loc='upper left')
                st.pyplot(fig)
            else:
                st.error("無法抓取該股票歷史數據。")
