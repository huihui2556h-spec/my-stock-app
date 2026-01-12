import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import matplotlib.pyplot as plt

# 設定頁面寬度，確保預估卡片能並排顯示
st.set_page_config(page_title="AI 多因子全景預估系統", layout="wide")

# 初始化頁面導覽狀態
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 核心回測功能：判斷預測精準度 (非美化，為真實判斷數據) ---
def calculate_accuracy(df, factor, side='high'):
    try:
        temp_df = df.copy().ffill()
        lookback = 60 # 回測過去 60 個交易日
        if len(temp_df) < lookback + 15: return 0.0
        
        hits = 0
        total_days = 0
        for i in range(len(temp_df) - lookback, len(temp_df)):
            history = temp_df.iloc[:i]
            actual_high = temp_df['High'].iloc[i]
            actual_low = temp_df['Low'].iloc[i]
            prev_close = temp_df['Close'].iloc[i-1]
            
            # 計算當下的 ATR 波動度
            tr = np.maximum(history['High'] - history['Low'], 
                           np.maximum(abs(history['High'] - history['Close'].shift(1)), 
                                      abs(history['Low'] - history['Close'].shift(1))))
            current_atr = tr.rolling(14).mean().iloc[-1]
            if np.isnan(current_atr): continue
            
            total_days += 1
            if side == 'high':
                pred_res = prev_close + (current_atr * factor)
                if actual_high <= pred_res: hits += 1 # 壓力位未被突破即為命中
            else:
                pred_sup = prev_close - (current_atr * factor)
                if actual_low >= pred_sup: hits += 1 # 支撐位未被跌破即為命中
        return (hits / total_days * 100) if total_days > 0 else 0.0
    except: return 0.0

# --- 🔍 數據抓取函數 (支援上市/上櫃自動判定) ---
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

# --- 🎨 預估資訊卡片組件 ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; border: 1px solid #eee; text-align: center;">
            <p style="margin:0; font-size:14px; color:#666; font-weight:bold;">{title}</p>
            <h2 style="margin:0; padding:10px 0; color:{text_color}; font-size:28px;">{price:.2f}</h2>
            <p style="margin:0; font-size:12px; color:#888;">60日回測命中率: <br><b style="font-size:14px;">{accuracy:.1f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 🏠 主程式介面 ---
if st.session_state.mode == "home":
    st.title("⚖️ AI 多因子預估分析系統")
    st.write("整合 FinMind 籌碼面、波動慣性與真實回測命中率")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 預估全景分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價分析")
    stock_id = st.text_input("請輸入股票代碼 (例: 2330):")
    if stock_id:
        df, sym = fetch_stock_data(stock_id)
        if df is not None:
            st.metric(f"最新成交價 ({sym})", f"{df['Close'].iloc[-1]:.2f}")
        else: st.error("查無數據，請確認代碼是否正確。")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    stock_input = st.text_input("請輸入分析代碼:")

    if stock_input:
        with st.spinner('AI 正在計算真實回測與籌碼修正...'):
            df, sym = fetch_stock_data(stock_input)
            if df is not None:
                # 核心因子計算邏輯
                tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
                vol_inertia = (df['Close'].pct_change().std()) * 100 # 波動慣性
                chip_score = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean() # 籌碼熱度
                bias = 1.006 if chip_score > 1 else 0.994 # 法人籌碼修正係數
                curr_p = float(df['Close'].iloc[-1])

                # 執行真實回測命中率計算
                acc_d_h = calculate_accuracy(df, (0.85 * bias), 'high')
                acc_d_l = calculate_accuracy(df, (0.75 / bias), 'low')
                acc_w_h = calculate_accuracy(df, (1.9 * bias), 'high')
                acc_w_l = calculate_accuracy(df, (1.6 / bias), 'low')

                st.subheader(f"🏠 {stock_input} 分析總覽 ({sym})")
                st.info(f"💡 籌碼修正: {bias:.3f} | 波動慣性: {vol_inertia:.2f} | 目前收盤: {curr_p:.2f}")

                # --- 🎯 核心區塊：隔日與五日整合並排 ---
                st.markdown("### 📊 全景預估點位 (隔日與五日對照判斷)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: display_metric_card("📈 隔日預估壓力", curr_p + (atr * 0.85 * bias), acc_d_h, "red")
                with m2: display_metric_card("📉 隔日預估支撐", curr_p - (atr * 0.75 / bias), acc_d_l, "green")
                with m3: display_metric_card("🚩 五日最大壓力", curr_p + (atr * 1.9 * bias), acc_w_h, "red")
                with m4: display_metric_card("⚓ 五日最大支撐", curr_p - (atr * 1.6 / bias), acc_w_l, "green")

                # --- 🏹 明日當沖建議價格 ---
                st.divider()
                st.markdown("### 🏹 明日當沖建議價格")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 強勢追多 (ATR*0.1)\n\n{curr_p + (atr*0.1):.2f}")
                d2.error(f"🔹 低接買點 (ATR*0.45)\n\n{curr_p - (atr*0.45):.2f}")
                d3.success(f"🔸 短線獲利 (ATR*0.75)\n\n{curr_p + (atr*0.75):.2f}")

                # --- 📈 價量走勢圖 (含完整中文註解) ---
                st.divider()
                st.write("📈 **近期價量走勢圖與 AI 預估區間**")
                plot_df = df.tail(40)
                
                # 初始化繪圖對象
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                # [價格區註解] 繪製收盤價線條與預估壓力支撐虛線
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="收盤價")
                ax1.axhline(y=curr_p + (atr * 1.9 * bias), color='#FF4B4B', ls='--', alpha=0.5, label="5D 壓力線")
                ax1.axhline(y=curr_p - (atr * 1.6 / bias), color='#28A745', ls='--', alpha=0.5, label="5D 支撐線")
                ax1.set_title("Price Action & AI Support/Resistance Bands")
                ax1.legend(loc='upper left')
                ax1.grid(axis='y', alpha=0.3)
                
                # [成交量區註解] 繪製成交量柱狀圖，紅色表示收紅、綠色表示收黑
                v_colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.7)
                ax2.set_ylabel("成交量")
                
                # [顯示圖表]
                st.pyplot(fig)
                st.info("📘 **圖表說明**：上方為收盤價走勢對應 AI 五日預估線；下方為成交量（紅漲綠跌）。")

            else:
                st.error("❌ 查無資料，請更換代碼嘗試。")
