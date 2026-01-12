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
# 1. 頁面配置
# =========================================================
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 籌碼因子：整合 FinMind 法人籌碼 ---
def get_chip_factor(stock_id):
    """計算法人籌碼權重修正因子"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取近 15 天數據
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            # 計算近五日買賣超淨額
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net_buy > 0:
                return 1.025, "✅ 籌碼面：法人偏多 (近五日買超)"
            else:
                return 0.975, "⚠️ 籌碼面：法人偏空 (近五日賣超)"
    except:
        pass
    return 1.0, "ℹ️ 籌碼面：中性 (數據同步中)"

# --- 🌍 國際局勢：美股 S&P 500 指數 ---
def get_international_bias():
    """美股對台股開盤影響因子"""
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 準確率回測邏輯 (60 日高精度) ---
def calculate_real_accuracy(df, atr_factor, chip_f=1.0, side='high'):
    """回測 60 個交易日的 AI 預估達成率"""
    try:
        df_copy = df.copy().ffill()
        if isinstance(df_copy.columns, pd.MultiIndex): df_copy.columns = df_copy.columns.get_level_values(0)
        backtest_days = min(len(df_copy) - 15, 60)
        if backtest_days <= 0: return 0.0
        hits = 0
        df_copy['ATR'] = (df_copy['High'] - df_copy['Low']).rolling(14).mean()
        
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = df_copy['ATR'].iloc[idx-1]
            if np.isnan(prev_atr): continue
            
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            pred_val = prev_close + (prev_atr * atr_factor * chip_f) if side == 'high' else prev_close - (prev_atr * atr_factor / chip_f)
            
            if side == 'high' and actual_val >= pred_val: hits += 1
            elif side == 'low' and actual_val <= pred_val: hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

# --- 獲取股票中文名稱 ---
def get_stock_name(stock_id):
    """從 Yahoo 財經抓取股票中文簡稱"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        # 使用正則表達式尋找標題中的名稱
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 視覺卡片組件 ---
def stock_box(label, price, pct, acc, color_type="red"):
    bg_color = "#FF4B4B" if color_type == "red" else "#28A745"
    arrow = "↑" if color_type == "red" else "↓"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {bg_color}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{bg_color}; color:white; padding:2px 8px; border-radius:5px; font-size:14px;">
                {arrow} {pct:.2f}%
            </span>
            <p style="margin-top:10px; font-size:12px; color:#888;">↳ 近 60 日 AI 達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# 2. 主程式介面
# =========================================================

if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("整合：國際局勢、量能慣性、**法人籌碼因子**、60日高精度回測")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日及波段預估分析")
    stock_id = st.text_input("輸入股票代碼 (例: 2330):")

    if stock_id:
        with st.spinner('AI 正在計算多因子模型...'):
            # 自動偵測上市/上櫃代碼
            df = None
            for suffix in [".TW", ".TWO"]:
                temp_df = yf.download(f"{stock_id}{suffix}", period="150d", progress=False)
                if not temp_df.empty:
                    df = temp_df
                    break
            
            if df is None or df.empty:
                st.error("❌ 找不到該代碼數據，請檢查輸入是否正確。")
                st.stop()

            # 抓取中文名稱
            stock_name = get_stock_name(stock_id)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.ffill()
            
            # 因子計算
            market_bias, market_pct = get_international_bias()
            chip_factor, chip_msg = get_chip_factor(stock_id)
            vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
            curr_vol = df['Volume'].iloc[-1]
            vol_factor = 1.05 if curr_vol > vol_ma5 else 0.95 

            # 核心邏輯 (ATR 波動率)
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            curr_c = float(df['Close'].iloc[-1])
            est_open = curr_c + (atr * 0.05 * market_bias)

            # 多因子預估點位
            total_bias = market_bias * chip_factor * vol_factor
            pred_h1 = curr_c + (atr * 0.85 * total_bias)
            pred_h5 = curr_c + (atr * 1.9 * total_bias)
            pred_l1 = curr_c - (atr * 0.65 / total_bias)
            pred_l5 = curr_c - (atr * 1.6 / total_bias)

            # 回測準確率
            acc_h1 = calculate_real_accuracy(df, 0.85, chip_f=chip_factor, side='high')
            acc_h5 = calculate_real_accuracy(df, 1.9, chip_f=chip_factor, side='high')
            acc_l1 = calculate_real_accuracy(df, 0.65, chip_f=chip_factor, side='low')
            acc_l5 = calculate_real_accuracy(df, 1.6, chip_f=chip_factor, side='low')

            # --- 畫面呈現 (中文回歸) ---
            st.subheader(f"🏠 {stock_name} ({stock_id})")
            st.write(f"🧬 **{chip_msg}**")
            
            m_color = "red" if market_pct < 0 else "green"
            st.write(f"🌍 **國際局勢參考 (美股 S&P 500)**: <span style='color:{m_color}'>{market_pct:+.2f}%</span>", unsafe_allow_html=True)
            
            v1, v2 = st.columns(2)
            v1.metric("目前收盤價", f"{curr_c:.2f}")
            v2.metric("預估明日開盤", f"{est_open:.2f}", delta=f"{est_open-curr_c:.2f}")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.write("🎯 **壓力預估 (多因子修正)**")
                stock_box("📈 隔日最高價", pred_h1, ((pred_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高價", pred_h5, ((pred_h5/curr_c)-1)*100, acc_h5, "red")
            with c2:
                st.write("🛡️ **支撐預估 (多因子修正)**")
                stock_box("📉 隔日最低價", pred_l1, ((pred_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低價", pred_l5, ((pred_l5/curr_c)-1)*100, acc_l5, "green")

            # --- 明日當沖建議 ---
            st.divider()
            st.markdown("### 🏹 明日當沖建議參考點位")
            d1, d2, d3 = st.columns(3)
            d1.info(f"🔹 強勢追多\n\n{est_open - (atr * 0.1 * vol_factor):.2f}")
            d2.error(f"🔹 低接買點\n\n{curr_c - (atr * 0.45 / market_bias):.2f}")
            d3.success(f"🔸 短線獲利\n\n{curr_c + (atr * 0.75 * market_bias):.2f}")

            # --- 📊 價量走勢圖 (中文化) ---
            st.divider()
            st.write(f"📈 **{stock_name} 近期價量走勢圖**")
            
            # 設定字體防止亂碼 (Streamlit Cloud 通常支援中文字體，若本地端報錯可移除 label)
            plot_df = df.tail(40).copy()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="收盤價")
            ax1.axhline(y=pred_h5, color='#FF4B4B', ls='--', alpha=0.5, label="AI 壓力線")
            ax1.axhline(y=pred_l5, color='#28A745', ls='--', alpha=0.5, label="AI 支撐線")
            ax1.set_ylabel("價格 (TWD)")
            ax1.legend(loc='upper left')
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_title(f"{stock_name} ({stock_id}) 歷史趨勢與 AI 點位", fontsize=14)

            # 成交量變色
            v_colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Close'].iloc[i-1] else 'green' for i in range(len(plot_df))]
            ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.7)
            ax2.set_ylabel("成交量")
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            st.info("📘 **圖表說明**：上方為收盤價走勢與 AI 壓力支撐線；下方為成交量（紅漲綠跌）。")
