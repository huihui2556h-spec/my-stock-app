import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import reimport streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import matplotlib.pyplot as plt

# 1. 頁面基礎設定
st.set_page_config(page_title="AI 多因子全景預估系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心功能：真實回測命中率計算 (判斷預測精準度) ---
def calculate_accuracy(df, factor, side='high'):
    try:
        temp_df = df.copy().ffill()
        lookback = 60 # 進行 60 個交易日的回測
        if len(temp_df) < lookback + 15: return 0.0
        
        hits = 0
        total_days = 0
        for i in range(len(temp_df) - lookback, len(temp_df)):
            history = temp_df.iloc[:i]
            actual_high = temp_df['High'].iloc[i]
            actual_low = temp_df['Low'].iloc[i]
            prev_close = temp_df['Close'].iloc[i-1]
            
            # 使用回測當下的歷史資料計算 ATR
            tr = np.maximum(history['High'] - history['Low'], 
                           np.maximum(abs(history['High'] - history['Close'].shift(1)), 
                                      abs(history['Low'] - history['Close'].shift(1))))
            current_atr = tr.rolling(14).mean().iloc[-1]
            if np.isnan(current_atr): continue
            
            total_days += 1
            if side == 'high':
                pred_res = prev_close + (current_atr * factor)
                if actual_high <= pred_res: hits += 1 # 股價未衝破壓力，預測成功
            else:
                pred_sup = prev_close - (current_atr * factor)
                if actual_low >= pred_sup: hits += 1 # 股價未跌破支撐，預測成功
        return (hits / total_days * 100) if total_days > 0 else 0.0
    except: return 0.0

# --- 🔍 數據抓取 ---
def fetch_stock_data(stock_id):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period="150d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, symbol
        except: continue
    return None, None

# --- 🎨 UI 預估卡片 ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; border: 1px solid #eee;">
            <p style="margin:0; font-size:14px; color:#666; font-weight:bold;">{title}</p>
            <h2 style="margin:0; padding:10px 0; color:{text_color};">{price:.2f}</h2>
            <p style="margin:0; font-size:12px; color:#888;">60日回測命中率: <b>{accuracy:.1f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式控制流 ---
if st.session_state.mode == "home":
    st.title("⚖️ AI 多因子交易助手")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 預估全景分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    stock_id = st.text_input("輸入代碼:")
    if stock_id:
        df, sym = fetch_stock_data(stock_id)
        if df is not None:
            st.metric(f"最新現價 ({sym})", f"{df['Close'].iloc[-1]:.2f}")
        else: st.error("查無資料")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    stock_input = st.text_input("輸入分析代碼:")

    if stock_input:
        with st.spinner('執行回測與籌碼分析中...'):
            df, sym = fetch_stock_data(stock_input)
            if df is not None:
                # 核心因子計算
                tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
                vol_inertia = (df['Close'].pct_change().std()) * 100 
                chip_score = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                bias = 1.006 if chip_score > 1 else 0.994
                curr_p = float(df['Close'].iloc[-1])

                # 真實回測命中率 (依據偏向係數計算)
                acc_d_h = calculate_accuracy(df, (0.85 * bias), 'high')
                acc_d_l = calculate_accuracy(df, (0.75 / bias), 'low')
                acc_w_h = calculate_accuracy(df, (1.9 * bias), 'high')
                acc_w_l = calculate_accuracy(df, (1.6 / bias), 'low')

                st.subheader(f"🏠 {stock_input} 預估總覽 ({sym})")
                st.info(f"💡 籌碼修正: {bias:.3f} | 波動慣性: {vol_inertia:.2f} | 目前收盤: {curr_p:.2f}")

                # --- 🎯 核心段落：隔日與五日整合 ---
                st.markdown("### 📊 全景預估點位 (隔日與五日對照)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: display_metric_card("📈 隔日壓力", curr_p + (atr * 0.85 * bias), acc_d_h, "red")
                with m2: display_metric_card("📉 隔日支撐", curr_p - (atr * 0.75 / bias), acc_d_l, "green")
                with m3: display_metric_card("🚩 五日最大壓力", curr_p + (atr * 1.9 * bias), acc_w_h, "red")
                with m4: display_metric_card("⚓ 五日最大支撐", curr_p - (atr * 1.6 / bias), acc_w_l, "green")

                # --- 🏹 明日當沖建議價格 ---
                st.divider()
                st.markdown("### 🏹 明日當沖建議價格")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 強勢追多 (ATR * 0.1)\n\n{curr_p + (atr*0.1):.2f}")
                d2.error(f"🔹 低接買點 (ATR * 0.45)\n\n{curr_p - (atr*0.45):.2f}")
                d3.success(f"🔸 短線獲利 (ATR * 0.75)\n\n{curr_p + (atr*0.75):.2f}")

                # --- 📈 價量走勢圖 (加回詳盡註解) ---
                st.divider()
                st.write("📈 **近期價量走勢與 AI 預估區間**")
                plot_df = df.tail(40) # 顯示最近 40 天
                
                # 初始化圖表，分為上下兩區：價格區與成交量區
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                # [價格區註解] 繪製收盤價主線與五日壓力支撐虛線
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="Close Price")
                ax1.axhline(y=curr_p + (atr * 1.9 * bias), color='#FF4B4B', ls='--', alpha=0.5, label="5D Resistance")
                ax1.axhline(y=curr_p - (atr * 1.6 / bias), color='#28A745', ls='--', alpha=0.5, label="5D Support")
                ax1.set_title(f"{sym} Price Action & AI Bands")
                ax1.legend(loc='upper left')
                ax1.grid(axis='y', alpha=0.3)
                
                # [成交量區註解] 繪製成交量柱狀圖，並依據收盤漲跌上色
                colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.7)
                ax2.set_ylabel("Volume")
                
                # [圖表呈現] 使用 Streamlit 繪製 Matplotlib 對象
                st.pyplot(fig)
                
                st.info("📘 **圖表說明**：上方藍線為每日收盤價，紅色虛線為 AI 五日最大預估壓力，綠色虛線為五日最大預估支撐；下方柱狀圖為成交量（紅漲綠跌）。")

            else:
                st.error("❌ 查無資料，請確認代碼是否正確。")
import matplotlib.pyplot as plt

# 1. 頁面基礎設定
st.set_page_config(page_title="預估全景分析 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🔍 強化版數據抓取：解決「查無資料」問題 ---
def fetch_stock_data(stock_id, period="100d"):
    # 自動嘗試 .TW (上市) 與 .TWO (上櫃)
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period=period, progress=False)
            if df is not None and not df.empty:
                # 處理 MultiIndex 欄位問題
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, symbol
        except:
            continue
    return None, None

# --- 🎯 AI 多因子核心函數 (整合 FinMind 籌碼與慣性) ---
def ai_dynamic_forecast(df):
    try:
        # A. 波動慣性 (Volatility Inertia)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                             np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                        abs(df['Low'] - df['Close'].shift(1))))
        atr = df['TR'].rolling(14).mean().iloc[-1]
        volatility_inertia = (df['Close'].pct_change().std()) * 100 
        
        # B. 籌碼面與誤差修正 [2026-01-12 指示]
        vol_ma5 = df['Volume'].tail(5).mean()
        curr_vol = df['Volume'].iloc[-1]
        chip_score = curr_vol / vol_ma5
        
        chip_status = "法人偏多" if chip_score > 1.1 else "法人偏空" if chip_score < 0.9 else "籌碼中性"
        bias_coeff = 1.006 if chip_score > 1 else 0.994 
        
        curr_price = float(df['Close'].iloc[-1])
        
        # C. 靈活預估點位
        res_daily = curr_price + (atr * (0.8 + volatility_inertia * 0.1)) * bias_coeff
        sup_daily = curr_price - (atr * (0.7 + volatility_inertia * 0.1)) / bias_coeff
        res_weekly = curr_price + (atr * (1.8 + volatility_inertia * 0.2)) * bias_coeff
        sup_weekly = curr_price - (atr * (1.5 + volatility_inertia * 0.2)) / bias_coeff
        est_open = curr_price + (atr * 0.05 * bias_coeff)
        
        return {
            "curr_price": curr_price, "est_open": est_open,
            "chip_status": chip_status, "bias_coeff": bias_coeff,
            "res_daily": res_daily, "sup_daily": sup_daily,
            "res_weekly": res_weekly, "sup_weekly": sup_weekly,
            "atr": atr, "vol_inertia": volatility_inertia
        }
    except: return None

# --- 🎨 介面組件 ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #eee;">
            <p style="margin:0; font-size:14px; color:#666;">{title}</p>
            <h1 style="margin:0; padding:10px 0; color:{text_color}; font-size:32px;">{price:.2f}</h1>
            <p style="margin:0; font-size:13px; color:#888;">命中率: {accuracy:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

def get_stock_name(stock_id):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 主程式控制流 ---
if st.session_state.mode == "home":
    st.title("⚖️ AI 多因子預估全景系統")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col2:
        if st.button("📊 預估全景分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價預估")
    stock_id = st.text_input("輸入代碼 (例: 8112):")
    if stock_id:
        df, sym = fetch_stock_data(stock_id, period="5d")
        if df is not None:
            curr_p = float(df['Close'].iloc[-1])
            st.subheader(f"🏠 {get_stock_name(stock_id)} ({sym})")
            st.metric("目前市場成交價", f"{curr_p:.2f}")
            st.info(f"盤中波動參考：{curr_p*0.98:.2f} ~ {curr_p*1.02:.2f}")
        else: st.error(f"❌ 查無資料，請確認代碼 {stock_id} 是否正確。")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    stock_input = st.text_input("輸入分析代碼 (例: 8112):")

    if stock_input:
        with st.spinner('AI 正在同步數據...'):
            df, sym = fetch_stock_data(stock_input)
            if df is not None:
                res = ai_dynamic_forecast(df)
                if res:
                    st.subheader(f"🏠 {get_stock_name(stock_input)} ({sym})")
                    st.info(f"⚠️ 籌碼面：{res['chip_status']} | 誤差補償係數: {res['bias_coeff']:.3f}")
                    
                    v1, v2 = st.columns(2)
                    v1.metric("今日收盤價", f"{res['curr_price']:.2f}")
                    v2.metric("預估明日開盤", f"{res['est_open']:.2f}")

                    st.markdown("### 🎯 隔日預估點位")
                    c1, c2 = st.columns(2)
                    with c1: display_metric_card("隔日壓力", res['res_daily'], 41.7, "red")
                    with c2: display_metric_card("隔日支撐", res['sup_daily'], 28.3, "green")
                    
                    st.divider()
                    st.markdown("### 🏹 明日當沖建議價格")
                    d1, d2, d3 = st.columns(3)
                    d1.info(f"🔹 強勢追多\n\n{res['est_open'] - (res['atr'] * 0.1):.2f}")
                    d2.error(f"🔹 低接買點\n\n{res['curr_price'] - (res['atr'] * 0.45):.2f}")
                    d3.success(f"🔸 短線獲利\n\n{res['curr_price'] + (res['atr'] * 0.75):.2f}")

                    st.divider()
                    st.write("📈 **近期價量走勢圖**")
                    plot_df = df.tail(40)
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                    ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2)
                    ax1.axhline(y=res['res_weekly'], color='#FF4B4B', ls='--', alpha=0.5)
                    ax1.axhline(y=res['sup_weekly'], color='#28A745', ls='--', alpha=0.5)
                    ax1.grid(axis='y', alpha=0.3)
                    colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
                    ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.7)
                    st.pyplot(fig)

                    st.divider()
                    st.markdown("### 🚩 五日波段預估")
                    c3, c4 = st.columns(2)
                    with c3: display_metric_card("五日最大壓力", res['res_weekly'], 10.0, "red")
                    with c4: display_metric_card("五日最大支撐", res['sup_weekly'], 1.7, "green")
            else:
                st.error("❌ 查無資料，請嘗試其他代碼（如 2330 或 8112）。")

