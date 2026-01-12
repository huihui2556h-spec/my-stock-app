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
# 1. 頁面配置 (還原您的視覺設定)
# =========================================================
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🎯 籌碼因子：整合 FinMind 法人籌碼 (新加入) ---
def get_chip_factor(stock_id):
    """獲取法人籌碼，計算權重修正因子"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取近 15 天數據以計算 5 日均值
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            # 計算近 5 個交易日法人買賣超淨額
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net_buy > 0:
                return 1.025, "✅ 籌碼面：法人偏多 (近五日買超)"
            else:
                return 0.975, "⚠️ 籌碼面：法人偏空 (近五日賣超)"
    except:
        pass
    return 1.0, "ℹ️ 籌碼面：中性 (FinMind 數據連線中)"

# --- 🌍 國際局勢：獲取美股 S&P 500 表現 ---
def get_international_bias():
    """連動美股 S&P 500 對台股開盤的影響"""
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        # 處理 yfinance 可能產生的 MultiIndex 欄位
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 核心準確率計算函數 (還原 60 日高精度回測) ---
def calculate_real_accuracy(df, atr_factor, chip_f=1.0, side='high'):
    """回測過去 60 個交易日，計算 AI 預估值的達成率"""
    try:
        df_copy = df.copy().ffill()
        if isinstance(df_copy.columns, pd.MultiIndex): df_copy.columns = df_copy.columns.get_level_values(0)
        backtest_days = min(len(df_copy) - 15, 60)
        if backtest_days <= 0: return 0.0
        hits = 0
        # 計算 ATR 波動率軌跡
        df_copy['ATR'] = (df_copy['High'] - df_copy['Low']).rolling(14).mean()
        
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = df_copy['ATR'].iloc[idx-1]
            if np.isnan(prev_atr): continue
            
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            # 模擬歷史預估點位 (加入籌碼修正)
            pred_val = prev_close + (prev_atr * atr_factor * chip_f) if side == 'high' else prev_close - (prev_atr * atr_factor / chip_f)
            
            # 判斷是否觸及 (最高價 > 預估價 或 最低價 < 預估價)
            if side == 'high' and actual_val >= pred_val: hits += 1
            elif side == 'low' and actual_val <= pred_val: hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

# --- 獲取中文名稱 ---
def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 自動偵測台股/上櫃代碼 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(stock_id, period="150d"):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period=period, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, symbol
        except: continue
    return None, None

# --- 🎨 視覺配色組件 (您原始的 HTML/CSS 卡片) ---
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
# 3. 主程式邏輯 (完整保留中文說明)
# =========================================================

if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("整合：國際局勢連動、量能籌碼修正、60日高精度回測、當沖策略指引")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日及波段預估")
    stock_id = st.text_input("輸入代碼 (如: 2330):")

    if stock_id:
        with st.spinner('正在分析多維度因子與回測數據...'):
            df, sym = fetch_stock_data(stock_id)
            if df is None or df.empty:
                st.error("❌ 找不到數據，請確認代碼。")
                st.stop()

            name = get_stock_name(stock_id)
            df = df.ffill()
            
            # 1. 因子獲取 (國際 + 量能 + FinMind 籌碼)
            market_bias, market_pct = get_international_bias()
            chip_factor, chip_msg = get_chip_factor(stock_id)
            
            # 量能因子：當日成交量 vs 5日均量
            vol_ma5 = df['Volume'].rolling(5).mean().iloc[-1]
            curr_vol = df['Volume'].iloc[-1]
            vol_factor = 1.05 if curr_vol > vol_ma5 else 0.95 

            # 2. 核心計算 (ATR 波動率)
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            curr_c = float(df['Close'].iloc[-1])
            
            # 預估明日開盤價 (考慮美股影響)
            est_open = curr_c + (atr * 0.05 * market_bias)

            # 3. 多因子合成預估
            # 綜合權重 = 國際盤偏向 * 籌碼因子 * 量能因子
            total_bias = market_bias * chip_factor * vol_factor
            
            pred_h1 = curr_c + (atr * 0.85 * total_bias)
            pred_h5 = curr_c + (atr * 1.9 * total_bias)
            pred_l1 = curr_c - (atr * 0.65 / total_bias)
            pred_l5 = curr_c - (atr * 1.6 / total_bias)

            # 4. 準確率回測
            acc_h1 = calculate_real_accuracy(df, 0.85, chip_f=chip_factor, side='high')
            acc_h5 = calculate_real_accuracy(df, 1.9, chip_f=chip_factor, side='high')
            acc_l1 = calculate_real_accuracy(df, 0.65, chip_f=chip_factor, side='low')
            acc_l5 = calculate_real_accuracy(df, 1.6, chip_f=chip_factor, side='low')

            # --- 畫面呈現 ---
            st.subheader(f"🏠 {name} ({stock_id})")
            st.write(f"🧬 **{chip_msg}**") # 顯示籌碼面註解
            
            m_color = "red" if market_pct < 0 else "green"
            st.write(f"🌍 **國際局勢參考 (S&P 500)**: <span style='color:{m_color}'>{market_pct:+.2f}%</span>", unsafe_allow_html=True)
            
            v1, v2 = st.columns(2)
            v1.metric("目前收盤價", f"{curr_c:.2f}")
            v2.metric("預估明日開盤", f"{est_open:.2f}", delta=f"{est_open-curr_c:.2f}")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.write("🎯 **壓力預估 (含多因子修正)**")
                stock_box("📈 隔日最高", pred_h1, ((pred_h1/curr_c)-1)*100, acc_h1, "red")
                stock_box("🚩 五日最高", pred_h5, ((pred_h5/curr_c)-1)*100, acc_h5, "red")
            with c2:
                st.write("🛡️ **支撐預估 (含多因子修正)**")
                stock_box("📉 隔日最低", pred_l1, ((pred_l1/curr_c)-1)*100, acc_l1, "green")
                stock_box("⚓ 五日最低", pred_l5, ((pred_l5/curr_c)-1)*100, acc_l5, "green")

            # --- 明日當沖建議價格 ---
            st.divider()
            st.markdown("### 🏹 明日當沖建議價格")
            d1, d2, d3 = st.columns(3)
            d1.info(f"🔹 強勢追多\n\n{est_open - (atr * 0.1 * vol_factor):.2f}")
            d2.error(f"🔹 低接買點\n\n{curr_c - (atr * 0.45 / market_bias):.2f}")
            d3.success(f"🔸 短線獲利\n\n{curr_c + (atr * 0.75 * market_bias):.2f}")

            # --- 📊 價量走勢圖 ---
            st.divider()
            st.write("📈 **近期價量走勢圖**")
            plot_df = df.tail(40).copy()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="收盤價 Price")
            ax1.axhline(y=pred_h5, color='#FF4B4B', ls='--', alpha=0.5, label="AI 壓力 Resistance")
            ax1.axhline(y=pred_l5, color='#28A745', ls='--', alpha=0.5, label="AI 支撐 Support")
            ax1.set_ylabel("價格 Price")
            ax1.legend(loc='upper left')
            ax1.grid(axis='y', alpha=0.3)
            
            # 彩色成交量 (紅漲綠跌)
            v_colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Close'].iloc[i-1] else 'green' for i in range(len(plot_df))]
            ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.7)
            ax2.set_ylabel("成交量 Volume")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.info("📘 **圖表說明**：上方為收盤價走勢與 AI 壓力支撐線；下方為成交量（紅漲綠跌）。")
            st.markdown(f"""
            * **達成率計算**：回測過去 **60 個交易日** 之歷史數據。
            * **主力進出修正**：根據成交量與 5 日均量關係調整敏感度。
            * **國際局勢**：連動 S&P 500 指數。
            * <span style="color:#FF4B4B">**Resistance (紅虛線)**</span>：預估五日最高壓力位。
            * <span style="color:#28A745">**Support (綠虛線)**</span>：預估五日最低支撐位。
            """, unsafe_allow_html=True)
