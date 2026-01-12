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
# 1. 系統環境設定 (設定網頁標籤與導航狀態)
# =========================================================
st.set_page_config(page_title="台股 AI 多因子交易系統 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    """【導航函數】處理頁面切換並重新渲染"""
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 核心運算引擎 (誤差補償 + FinMind 籌碼因子)
# =========================================================

def get_error_bias(df, days=10):
    """【誤差補償】計算過去10天AI預估偏離率，用來動態修正今日點位"""
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
            return (1.025, "✅ 籌碼面：法人偏多") if net_buy > 0 else (0.975, "⚠️ 籌碼面：法人偏空")
    except: pass
    return 1.0, "ℹ️ 籌碼面：中性數據"

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
# 3. 介面呈現 (首頁 / 盤中 / 深度預估)
# =========================================================

# --- 🏠 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("已整合：盤中監控、FinMind 籌碼、高精度誤差修正")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

# --- ⚡ 盤中即時頁面 (增加時間判斷與隱藏邏輯) ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時監控")
    
    # 判斷台灣交易時間
    tw_tz = pytz.timezone('Asia/Taipei')
    now_tw = datetime.datetime.now(tw_tz)
    is_trading_time = now_tw.weekday() < 5 and (datetime.time(9, 0) <= now_tw.time() <= datetime.time(13, 35))
    
    if not is_trading_time:
        st.warning(f"目前為非交易時段（現在時間：{now_tw.strftime('%H:%M:%S')}）")
        st.info("盤中即時數據僅在週一至週五 09:00 - 13:35 顯示。")
    else:
        sid_rt = st.text_input("請輸入股票代碼 (例: 2330):", key="rt_id")
        if sid_rt:
            with st.spinner('連線即時行情...'):
                df_rt = yf.download(f"{sid_rt}.TW", period="1d", interval="1m", progress=False)
                if df_rt.empty: df_rt = yf.download(f"{sid_rt}.TWO", period="1d", interval="1m", progress=False)
                
                if not df_rt.empty:
                    if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
                    st.subheader(f"🏠 {get_stock_name(sid_rt)} ({sid_rt})")
                    curr_p = df_rt['Close'].iloc[-1]
                    st.metric("即時成交價", f"{curr_p:.2f}")
                    st.line_chart(df_rt['Close'])
                else:
                    st.error("查無此代碼，請確認代號正確。")

# --- 📊 深度預估頁面 (垂直佈局 + 找回圖表) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    sid_fc = st.text_input("輸入分析代碼 (例: 2330):", key="fc_id")

    if sid_fc:
        with st.spinner('執行 AI 多因子運算中...'):
            df = None
            for suf in [".TW", ".TWO"]:
                tmp = yf.download(f"{sid_fc}{suf}", period="200d", progress=False)
                if not tmp.empty: df = tmp; break
            
            if df is not None:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # 計算運算權重
                chip_f, chip_msg = get_chip_factor(sid_fc)
                err_f = get_error_bias(df)
                total_f = chip_f * err_f
                curr_c = float(df['Close'].iloc[-1])
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                
                # 介面資訊
                st.subheader(f"🏠 {get_stock_name(sid_fc)} ({sid_fc})")
                st.info(f"{chip_msg} | 誤差補償係數: {err_f:.3f}")

                # 🎯 隔日預估區塊 (取消分頁，垂直排列)
                st.markdown("---")
                st.markdown("### 🎯 隔日預估點位")
                ph1, pl1 = curr_c + (atr * 0.85 * total_f), curr_c - (atr * 0.65 / total_f)
                c1, c2 = st.columns(2)
                c1.error(f"**📈 隔日壓力** \n## {ph1:.2f} \n<small>命中率: {calculate_real_accuracy(df, 0.85, 'high'):.1f}%</small>")
                c2.success(f"**📉 隔日支撐** \n## {pl1:.2f} \n<small>命中率: {calculate_real_accuracy(df, 0.65, 'low'):.1f}%</small>")
                
                # 🚩 五日預估區塊
                st.markdown("---")
                st.markdown("### 🚩 五日波段預估")
                ph5, pl5 = curr_c + (atr * 1.90 * total_f), curr_c - (atr * 1.60 / total_f)
                c3, c4 = st.columns(2)
                c3.error(f"**📈 五日最大壓力** \n## {ph5:.2f} \n<small>命中率: {calculate_real_accuracy(df, 1.90, 'high'):.1f}%</small>")
                c4.success(f"**📉 五日最大支撐** \n## {pl5:.2f} \n<small>命中率: {calculate_real_accuracy(df, 1.60, 'low'):.1f}%</small>")

                # --- 📊 找回圖表區塊 ---
                st.markdown("---")
                st.write("📊 歷史價量與 AI 預估區間圖 (Price Action Chart)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.tail(40).index, df.tail(40)['Close'], label="Price", color='#1f77b4', lw=2)
                # 繪製五日壓力與支撐線
                ax.axhline(y=ph5, color='red', ls='--', alpha=0.5, label="5D Resistance")
                ax.axhline(y=pl5, color='green', ls='--', alpha=0.5, label="5D Support")
                ax.legend(loc='upper left')
                st.pyplot(fig)
                st.caption("註：紅線為五日預估壓力，綠線為五日預估支撐。")
            else:
                st.error("無法抓取歷史數據。")
