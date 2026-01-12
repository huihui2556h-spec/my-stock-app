import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta
import pytz
from FinMind.data import DataLoader 

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🧬 外部資料庫：籌碼面分析模組 ---
def get_external_chip_factor(stock_id):
    try:
        dl = DataLoader()
        # 抓取近 10 天資料以計算近期趨勢
        start_dt = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_dt)
        
        chip_score = 1.0 
        if not inst_df.empty:
            recent_inst = inst_df.tail(9) 
            net_buy = recent_inst['buy'].sum() - recent_inst['sell'].sum()
            # 法人籌碼惯性修正
            if net_buy > 0: chip_score += 0.006 
            else: chip_score -= 0.006
            
        if not margin_df.empty:
            recent_margin = margin_df.tail(3)
            # 融資減、券增 通常視為籌碼集中
            if recent_margin['Margin_Purchase_today_balance'].iloc[-1] < recent_margin['Margin_Purchase_today_balance'].iloc[0]:
                chip_score += 0.003
        
        return chip_score
    except:
        return 1.0 

# --- 🧠 AI 動態特徵預測核心 (考慮波動慣性與法人籌碼) ---
def ai_dynamic_forecast(df, chip_factor=1.0):
    try:
        df_clean = df.tail(60).copy()
        df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        
        # 學習各股特有分位數波動 + 籌碼校正
        h1_dynamic = df_clean['h_pct'].quantile(0.75) * chip_factor
        h5_dynamic = df_clean['h_pct'].quantile(0.95) * chip_factor
        l1_dynamic = df_clean['l_pct'].quantile(0.25) / chip_factor
        l5_dynamic = df_clean['l_pct'].quantile(0.05) / chip_factor
        
        return h1_dynamic, h5_dynamic, l1_dynamic, l5_dynamic
    except:
        return 0.02, 0.05, -0.015, -0.04

def calculate_real_accuracy(df, target_pct, side='high'):
    try:
        df_copy = df.copy().tail(60)
        hits = 0
        for i in range(1, len(df_copy)):
            prev_c = df_copy['Close'].iloc[i-1]
            actual_val = df_copy['High'].iloc[i] if side == 'high' else df_copy['Low'].iloc[i]
            pred_val = prev_c * (1 + target_pct)
            if side == 'high' and actual_val >= pred_val: hits += 1
            elif side == 'low' and actual_val <= pred_val: hits += 1
        return (hits / len(df_copy)) * 100
    except: return 0.0

def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"股票 {stock_id}"

# --- 🎨 視覺組件 (保留原本紅綠設計) ---
def stock_box(label, price, pct, acc, color_type="red"):
    bg_color = "#FF4B4B" if color_type == "red" else "#28A745"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {bg_color}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{bg_color}; color:white; padding:2px 8px; border-radius:5px; font-size:13px;">
                預估振幅：{pct:.2f}%
            </span>
            <p style="margin-top:10px; font-size:11px; color:#888;">↳ 歷史特徵達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 主程式控制 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 隔日深度預估", use_container_width=True): navigate_to("forecast")

# =========================================================
# ⚡ 盤中即時 (保留重整功能)
# =========================================================
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    header_col, refresh_col = st.columns([4, 1.2])
    with header_col: st.title("⚡ 盤中動態決策")
    with refresh_col:
        st.write("") 
        if st.button("🔄 點擊重整", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    stock_id = st.text_input("輸入代碼:", key="rt_id_input")
    if stock_id:
        success = False
        for suffix in [".TW", ".TWO"]:
            symbol = f"{stock_id}{suffix}"
            df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
            if not df_rt.empty:
                success = True
                break
        
        if success:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            curr_p = float(df_rt['Close'].iloc[-1])
            vwap_p = float(df_rt['VWAP'].iloc[-1])
            st.subheader(f"🎯 {get_stock_name(stock_id)}")
            st.metric("即時現價", f"{curr_p:.2f}")
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.success(f"🔹 即時支撐 (VWAP)：{vwap_p:.2f}")
            c2.error(f"🔸 即時分批停利：{curr_p * 1.015:.2f}")
        else:
            st.warning("⚠️ 查無即時資料。")

# =========================================================
# 📊 隔日深度預估 (完整包含五日預測 + 籌碼因子)
# =========================================================
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日多因子 AI 預判")
    stock_id = st.text_input("輸入代碼:", key="fc_id_input")
    if stock_id:
        with st.spinner('正在分析各股波動慣性與法人籌碼...'):
            success = False
            for suffix in [".TW", ".TWO"]:
                symbol = f"{stock_id}{suffix}"
                df = yf.download(symbol, period="100d", progress=False)
                if not df.empty:
                    success = True
                    break
            
            if success:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                curr_c = float(df['Close'].iloc[-1])
                
                # 1. 籌碼因子校正
                chip_f = get_external_chip_factor(stock_id)
                
                # 2. AI 特徵預測 (包含五日最高低)
                h1_p, h5_p, l1_p, l5_p = ai_dynamic_forecast(df, chip_factor=chip_f)
                
                # 點位計算
                p_h1, p_h5 = curr_c * (1 + h1_p), curr_c * (1 + h5_p)
                p_l1, p_l5 = curr_c * (1 + l1_p), curr_c * (1 + l5_p)
                
                # 達成率回測
                acc_h1 = calculate_real_accuracy(df, h1_p, 'high')
                acc_h5 = calculate_real_accuracy(df, h5_p, 'high')
                acc_l1 = calculate_real_accuracy(df, l1_p, 'low')
                acc_l5 = calculate_real_accuracy(df, l5_p, 'low')

                st.subheader(f"🏠 {get_stock_name(stock_id)}")
                st.metric("最新收盤價", f"{curr_c:.2f}")
                
                # 顯示籌碼信心
                status = "🔥 籌碼強勢 (法人買超)" if chip_f > 1.0 else "❄️ 籌碼中性/弱勢"
                st.write(f"**AI 綜合分析：{status} (修正因子: {chip_f:.3f})**")
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    stock_box("📈 隔日最高預估", p_h1, h1_p*100, acc_h1, "red")
                    stock_box("🚩 五日最高預估", p_h5, h5_p*100, acc_h5, "red")
                with col2:
                    stock_box("📉 隔日最低預估", p_l1, l1_p*100, acc_l1, "green")
                    stock_box("⚓ 五日最低預估", p_l5, l5_p*100, acc_l5, "green")

                # 圖表
                st.divider()
                st.write("### 📉 歷史走勢與量價動能")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
                ax1.plot(df.index[-40:], df['Close'].tail(40), color='#1f77b4', lw=2)
                ax1.axhline(y=p_h1, color='red', ls='--', alpha=0.4, label="AI Resistance")
                ax1.axhline(y=p_l1, color='green', ls='--', alpha=0.4, label="AI Support")
                ax1.legend()
                
                plot_df = df.tail(40)
                colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.6)
                st.pyplot(fig)

                st.info("📘 **多因子 AI 說明**")
                st.markdown("""
                * **五日預估**：基於該股 95% 波動分位數與籌碼權重，預測未來一週可能的極端價格區間。
                * **籌碼修正**：自動考慮 **FinMind** 提供之法人買賣超與融資數據，動態調整支撐壓力位。
                """)
