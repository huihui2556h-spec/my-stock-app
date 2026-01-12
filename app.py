import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import matplotlib.pyplot as plt
import matplotlib

# --- 0. 設置中文字體 (解決圖片亂碼) ---
matplotlib.rc('font', family='Microsoft JhengHei' if 'Win' in str(matplotlib.get_backend()) else 'sans-serif')
plt.rcParams['axes.unicode_minus'] = False 

# 頁面配置
st.set_page_config(page_title="AI 全景預估分析", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心回測功能 ---
def calculate_accuracy(df, factor, side='high'):
    try:
        temp_df = df.copy().ffill()
        lookback = 60
        if len(temp_df) < lookback + 15: return 0.0
        hits, total_days = 0, 0
        for i in range(len(temp_df) - lookback, len(temp_df)):
            history = temp_df.iloc[:i]
            actual_high, actual_low = temp_df['High'].iloc[i], temp_df['Low'].iloc[i]
            prev_close = temp_df['Close'].iloc[i-1]
            tr = np.maximum(history['High'] - history['Low'], 
                           np.maximum(abs(history['High'] - history['Close'].shift(1)), 
                                      abs(history['Low'] - history['Close'].shift(1))))
            current_atr = tr.rolling(14).mean().iloc[-1]
            if np.isnan(current_atr): continue
            total_days += 1
            if side == 'high':
                if actual_high <= (prev_close + (current_atr * factor)): hits += 1
            else:
                if actual_low >= (prev_close - (current_atr * factor)): hits += 1
        return (hits / total_days * 100) if total_days > 0 else 0.0
    except: return 0.0

# --- 🔍 強化名稱與數據抓取 ---
def get_stock_info(stock_id):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period="150d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # 抓取名稱
                url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
                res = requests.get(url, timeout=5)
                name = re.search(r'<title>(.*?) \(', res.text).group(1).split('-')[0].strip()
                return df, symbol, name
        except: continue
    return None, None, f"台股 {stock_id}"

# --- 🎨 大字體 UI 組件 ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 20px; border-radius: 12px; border: 1px solid #ddd; text-align: center; margin-bottom: 10px;">
            <p style="margin:0; font-size:18px; color:#444; font-weight:bold;">{title}</p>
            <h2 style="margin:0; padding:10px 0; color:{text_color}; font-size:36px;">{price:.2f}</h2>
            <p style="margin:0; font-size:14px; color:#777;">回測命中率: <b style="font-size:16px; color:#333;">{accuracy:.1f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式 ---
if st.session_state.mode == "home":
    st.title("⚖️ AI 多因子預估全景系統")
    st.write("### 請選擇功能模式")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價 (開盤時段)", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 預估全景分析 (盤後/盤前建議)", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價")
    
    # 判定交易時段
    now = datetime.now()
    is_trading = (now.weekday() < 5) and (9 <= now.hour < 14)
    if not is_trading:
        st.warning(f"🕒 目前非交易時段 (現在時間: {now.strftime('%H:%M')})，顯示為最近收盤行情。")
    
    sid = st.text_input("輸入股票代碼 (例: 2330):")
    if sid:
        df, sym, name = get_stock_info(sid)
        if df is not None:
            st.markdown(f"<h1 style='font-size:45px;'>{name} <small style='color:gray;'>({sym})</small></h1>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("當前/最後成交價", f"{df['Close'].iloc[-1]:.2f}")
            col2.metric("當日最高", f"{df['High'].iloc[-1]:.2f}")
            col3.metric("當日最低", f"{df['Low'].iloc[-1]:.2f}")
        else: st.error("查無資料")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    stock_input = st.text_input("輸入代碼進行 60 日勝率回測:")

    if stock_input:
        with st.spinner('正在分析數據與回測勝率...'):
            df, sym, name = get_stock_info(stock_input)
            if df is not None:
                # 計算參數
                tr = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
                chip_score = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                bias = 1.006 if chip_score > 1 else 0.994
                curr_p = float(df['Close'].iloc[-1])

                # 真實回測
                acc_dh = calculate_accuracy(df, (0.85*bias), 'high')
                acc_dl = calculate_accuracy(df, (0.75/bias), 'low')
                acc_wh = calculate_accuracy(df, (1.9*bias), 'high')
                acc_wl = calculate_accuracy(df, (1.6/bias), 'low')

                # 1. 頂部核心：股票名稱與大字體收盤價
                st.divider()
                h1, h2 = st.columns([3, 2])
                with h1:
                    st.markdown(f"<h1 style='color:#1E3A8A; font-size:50px; margin-bottom:0;'>{name} <span style='font-size:30px; color:#666;'>({sym})</span></h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:#444; font-size:24px; margin:0;'>今日收盤價：<b style='font-size:70px; color:#C53030;'>{curr_p:.2f}</b></p>", unsafe_allow_html=True)
                with h2:
                    st.info(f"📈 籌碼修正：{bias:.3f} ({'偏多' if bias > 1 else '偏空'})\n\n🚩 波動慣性：{(df['Close'].pct_change().std()*100):.2f}\n\n🌅 預估明日開盤：{curr_p + (atr*0.05*bias):.2f}")

                # 2. 全景預估：隔日與五日
                st.markdown("### 🎯 隔日與五日預估對照 (60日回測)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: display_metric_card("📈 隔日壓力", curr_p + (atr*0.85*bias), acc_dh, "red")
                with m2: display_metric_card("📉 隔日支撐", curr_p - (atr*0.75/bias), acc_dl, "green")
                with m3: display_metric_card("🚩 五日最大壓力", curr_p + (atr*1.9*bias), acc_wh, "red")
                with m4: display_metric_card("⚓ 五日最大支撐", curr_p - (atr*1.6/bias), acc_wl, "green")

                # 3. 🏹 明日當沖建議價格
                st.divider()
                st.markdown("### 🏹 明日當沖與隔日預估建議")
                d1, d2, d3 = st.columns(3)
                # 使用大標籤顯示
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; gap: 10px;">
                    <div style="flex:1; background:#EBF8FF; padding:15px; border-radius:10px; border-left:5px solid #3182CE;">
                        <b style="color:#2C5282; font-size:18px;">🔹 強勢追多</b><br><span style="font-size:32px; font-weight:bold;">{curr_p + (atr*0.1):.2f}</span>
                    </div>
                    <div style="flex:1; background:#FFF5F5; padding:15px; border-radius:10px; border-left:5px solid #E53E3E;">
                        <b style="color:#9B2C2C; font-size:18px;">🔹 低接買點</b><br><span style="font-size:32px; font-weight:bold;">{curr_p - (atr*0.45):.2f}</span>
                    </div>
                    <div style="flex:1; background:#F0FFF4; padding:15px; border-radius:10px; border-left:5px solid #38A169;">
                        <b style="color:#22543D; font-size:18px;">🔸 短線獲利</b><br><span style="font-size:32px; font-weight:bold;">{curr_p + (atr*0.75):.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 4. 價量走勢圖 (含中文註解)
                st.divider()
                st.markdown("### 📈 近期價量走勢與 AI 預估軌跡")
                plot_df = df.tail(40)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=3, label="每日收盤價")
                ax1.axhline(y=curr_p + (atr*1.9*bias), color='#FF4B4B', ls='--', lw=2, alpha=0.6, label="五日預估壓力")
                ax1.axhline(y=curr_p - (atr*1.6/bias), color='#28A745', ls='--', lw=2, alpha=0.6, label="五日預估支撐")
                ax1.set_ylabel("價格", fontsize=14)
                ax1.legend(loc='upper left', fontsize=12)
                ax1.grid(alpha=0.3)
                
                v_colors = ['#EF5350' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else '#26A69A' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.8)
                ax2.set_ylabel("成交量", fontsize=14)
                
                st.pyplot(fig)
                st.info(f"📘 **圖表註解**：{name} 的紅/綠虛線代表 AI 根據目前籌碼係數 {bias:.3f} 算出的波段極限。")
            else:
                st.error("查無資料")
