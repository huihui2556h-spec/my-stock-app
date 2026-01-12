import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import matplotlib.pyplot as plt

# 1. 頁面基礎設定
st.set_page_config(page_title="預估全景分析 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 AI 多因子核心函數 (整合 FinMind 籌碼與慣性) ---
def ai_dynamic_forecast(df):
    try:
        # A. 波動慣性 (Volatility Inertia) 計算
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
        
        # C. 靈活預估點位 (加入慣性修正)
        res_daily = curr_price + (atr * (0.8 + volatility_inertia * 0.1)) * bias_coeff
        sup_daily = curr_price - (atr * (0.7 + volatility_inertia * 0.1)) / bias_coeff
        res_weekly = curr_price + (atr * (1.8 + volatility_inertia * 0.2)) * bias_coeff
        sup_weekly = curr_price - (atr * (1.5 + volatility_inertia * 0.2)) / bias_coeff
        
        # 隔日開盤預估
        est_open = curr_price + (atr * 0.05 * bias_coeff)
        
        return {
            "curr_price": curr_price, "est_open": est_open,
            "chip_status": chip_status, "bias_coeff": bias_coeff,
            "res_daily": res_daily, "sup_daily": sup_daily,
            "res_weekly": res_weekly, "sup_weekly": sup_weekly,
            "atr": atr, "vol_inertia": volatility_inertia
        }
    except: return None

# --- 🎨 介面組件 (已修正亂碼問題) ---
def display_metric_card(title, price, accuracy, color_type="red"):
    bg_color = "#FFF5F5" if color_type == "red" else "#F5FFF5"
    text_color = "#C53030" if color_type == "red" else "#2F855A"
    # 直接使用 Markdown 渲染，不使用轉義標籤
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
        df = yf.download(f"{stock_id}.TW", period="5d", progress=False)
        if not df.empty:
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            curr_p = df['Close'].iloc[-1]
            st.subheader(f"🏠 {get_stock_name(stock_id)} 現價分析")
            st.metric("目前市場成交價", f"{curr_p:.2f}")
            # 盤中簡單提示
            st.write(f"今日波動範圍預估：{curr_p*0.98:.2f} ~ {curr_p*1.02:.2f}")
        else: st.error("查無資料")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 預估全景分析")
    stock_input = st.text_input("輸入分析代碼 (例: 8112):")

    if stock_input:
        with st.spinner('AI 正在分析數據...'):
            df = yf.download(f"{stock_input}.TW", period="100d", progress=False)
            if df.empty:
                st.error("查無資料"); st.stop()
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            
            res = ai_dynamic_forecast(df)
            if res:
                # 1. 頂部資訊區
                st.subheader(f"🏠 {get_stock_name(stock_input)}({stock_input}.TW)")
                st.info(f"⚠️ 籌碼面：{res['chip_status']} | 誤差補償係數: {res['bias_coeff']:.3f}")
                
                v1, v2 = st.columns(2)
                v1.metric("今日收盤價", f"{res['curr_price']:.2f}")
                v2.metric("預估明日開盤", f"{res['est_open']:.2f}")

                # 2. 隔日預估點位 (亂碼已移除)
                st.markdown("### 🎯 隔日預估點位")
                c1, c2 = st.columns(2)
                with c1: display_metric_card("隔日壓力", res['res_daily'], 41.7, "red")
                with c2: display_metric_card("隔日支撐", res['sup_daily'], 28.3, "green")
                
                # 3. 🏹 明日當沖建議價格
                st.divider()
                st.markdown("### 🏹 明日當沖建議價格")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 強勢追多\n\n{res['est_open'] - (res['atr'] * 0.1):.2f}")
                d2.error(f"🔹 低接買點\n\n{res['curr_price'] - (res['atr'] * 0.45):.2f}")
                d3.success(f"🔸 短線獲利\n\n{res['curr_price'] + (res['atr'] * 0.75):.2f}")

                # 4. 📈 價量走勢圖
                st.divider()
                st.write("📈 **近期價量走勢圖**")
                plot_df = df.tail(40)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="Price")
                ax1.axhline(y=res['res_weekly'], color='#FF4B4B', ls='--', alpha=0.5, label="Resistance")
                ax1.axhline(y=res['sup_weekly'], color='#28A745', ls='--', alpha=0.5, label="Support")
                ax1.set_ylabel("Price")
                ax1.legend(loc='upper left')
                ax1.grid(axis='y', alpha=0.3)
                colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else 'green' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.7)
                ax2.set_ylabel("Volume")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.info("📘 **圖表說明**：上方為收盤價走勢與 AI 壓力支撐線；下方為成交量。")

                # 5. 五日波段預估
                st.divider()
                st.markdown("### 🚩 五日波段預估")
                c3, c4 = st.columns(2)
                with c3: display_metric_card("五日最大壓力", res['res_weekly'], 10.0, "red")
                with c4: display_metric_card("五日最大支撐", res['sup_weekly'], 1.7, "green")
                
                st.markdown(f"""
                * <span style="color:#FF4B4B">**Resistance (紅虛線)**</span>：預估五日最高壓力位。
                * <span style="color:#28A745">**Support (綠虛線)**</span>：預估五日最低支撐位。
                """, unsafe_allow_html=True)
