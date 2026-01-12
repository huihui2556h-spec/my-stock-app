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
# 1. 系統視覺與初始化
# =========================================================
st.set_page_config(page_title="台股 AI 多因子波段助手", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 核心運算引擎 (多因子修正 + 多維度時間預估)
# =========================================================

# --- 🎯 籌碼面：FinMind 法人籌碼權重 ---
def get_chip_factor(stock_id):
    """計算法人買賣超權重修正：考量法人近 5 日籌碼去向"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            # 合計三大法人買賣淨額
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net_buy > 0:
                return 1.025, "✅ 籌碼面：法人偏多 (近五日買超)"
            else:
                return 0.975, "⚠️ 籌碼面：法人偏空 (近五日賣超)"
    except: pass
    return 1.0, "ℹ️ 籌碼面：中性 (數據連線中)"

# --- 🌍 國際面：美股連動因子 ---
def get_international_bias():
    """計算美股昨日表現對台股開盤的加權影響"""
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        return 1 + (float(change) * 0.5), float(change) * 100
    except: return 1.0, 0.0

# --- 🎯 準確率回測：60日歷史回測 ---
def calculate_accuracy(df, atr_factor, chip_f=1.0, side='high'):
    """回測過去 60 天，預估點位被觸及的機率 (達成率)"""
    try:
        df_copy = df.copy().ffill()
        if isinstance(df_copy.columns, pd.MultiIndex): df_copy.columns = df_copy.columns.get_level_values(0)
        backtest_days = min(len(df_copy) - 15, 60)
        hits = 0
        df_copy['ATR'] = (df_copy['High'] - df_copy['Low']).rolling(14).mean()
        
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = df_copy['ATR'].iloc[idx-1]
            if np.isnan(prev_atr): continue
            
            actual = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            # 模擬 AI 預估值
            pred = prev_close + (prev_atr * atr_factor * chip_f) if side == 'high' else prev_close - (prev_atr * atr_factor / chip_f)
            
            if side == 'high' and actual >= pred: hits += 1
            elif side == 'low' and actual <= pred: hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

# --- 🔍 名稱抓取 ---
def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 🎨 視覺卡片組件 ---
def stock_box(label, price, pct, acc, color_type="red"):
    bg_color = "#FF4B4B" if color_type == "red" else "#28A745"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {bg_color}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{bg_color}; color:white; padding:2px 8px; border-radius:5px; font-size:14px;">
                {pct:+.2f}%
            </span>
            <p style="margin-top:10px; font-size:12px; color:#888;">↳ AI 達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# 3. 頁面邏輯
# =========================================================

if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("已整合：FinMind 籌碼因子、1/5/10 日多維度預估、美股連動影響")
    c_a, c_b = st.columns(2)
    with c_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c_b:
        if st.button("📊 深度預估分析", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 多維度波段深度預估")
    fc_id = st.text_input("輸入股票代碼 (如: 2330):", key="fc_in")

    if fc_id:
        with st.spinner('正在分析多因子數據與長線慣性...'):
            # 抓取數據
            df = None
            for suffix in [".TW", ".TWO"]:
                temp = yf.download(f"{fc_id}{suffix}", period="200d", progress=False)
                if not temp.empty:
                    df = temp
                    break
            
            if df is not None:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                name = get_stock_name(fc_id)
                curr_c = float(df['Close'].iloc[-1])
                
                # 因子獲取
                market_f, market_pct = get_international_bias()
                chip_f, chip_msg = get_chip_factor(fc_id)
                vol_f = 1.05 if df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] else 0.95
                total_bias = market_f * chip_f * vol_f
                
                # ATR 波動基準
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]

                # --- 核心計算：1/5/10 日預估點位 ---
                # 係數：0.85 (隔日), 1.9 (五日), 2.8 (十日)
                ph1, pl1 = curr_c + (atr * 0.85 * total_bias), curr_c - (atr * 0.65 / total_bias)
                ph5, pl5 = curr_c + (atr * 1.90 * total_bias), curr_c - (atr * 1.60 / total_bias)
                ph10, pl10 = curr_c + (atr * 2.80 * total_bias), curr_c - (atr * 2.30 / total_bias)

                # 回測準確率計算
                ah1 = calculate_accuracy(df, 0.85, chip_f, 'high')
                al1 = calculate_accuracy(df, 0.65, chip_f, 'low')
                ah5 = calculate_accuracy(df, 1.90, chip_f, 'high')
                al5 = calculate_accuracy(df, 1.60, chip_f, 'low')
                ah10 = calculate_accuracy(df, 2.80, chip_f, 'high')
                al10 = calculate_accuracy(df, 2.30, chip_f, 'low')

                # --- 介面呈現 ---
                st.subheader(f"🏠 {name} ({fc_id}) - 多維度預測")
                st.info(chip_msg)
                st.write(f"🌍 **美股連動參考**: {market_pct:+.2f}%")

                # 分別顯示 1, 5, 10 日預估
                tab1, tab5, tab10 = st.tabs(["🎯 隔日預估", "🚩 五日波段", "⚓ 十日長波段"])
                
                with tab1:
                    c1, c2 = st.columns(2)
                    with c1: stock_box("📈 隔日最高預估", ph1, ((ph1/curr_c)-1)*100, ah1, "red")
                    with c2: stock_box("📉 隔日最低預估", pl1, ((pl1/curr_c)-1)*100, al1, "green")
                
                with tab5:
                    c1, c2 = st.columns(2)
                    with c1: stock_box("📈 五日最高預估", ph5, ((ph5/curr_c)-1)*100, ah5, "red")
                    with c2: stock_box("📉 五日最低預估", pl5, ((pl5/curr_c)-1)*100, al5, "green")
                
                with tab10:
                    c1, c2 = st.columns(2)
                    with c1: stock_box("📈 十日最高預估", ph10, ((ph10/curr_c)-1)*100, ah10, "red")
                    with c2: stock_box("📉 十日最低預估", pl10, ((pl10/curr_c)-1)*100, al10, "green")

                # --- 🏹 明日當沖建議價格 ---
                st.divider()
                st.markdown("### 🏹 明日當沖建議參考點位")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 **強勢追多買點**\n\n**{curr_c + (atr * 0.1 * vol_f):.2f}**")
                d2.error(f"🔹 **回測支撐低階**\n\n**{curr_c - (atr * 0.45 / market_f):.2f}**")
                d3.success(f"🔸 **短線分批停利**\n\n**{curr_c + (atr * 0.75 * total_bias):.2f}**")

                # --- 📊 圖表：多維度視覺化 (無亂碼) ---
                st.divider()
                st.write(f"📊 **{name} 多維度壓力支撐圖 (1/5/10 Day)**")
                fig, ax = plt.subplots(figsize=(10, 5))
                plot_df = df.tail(50)
                ax.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="Price")
                
                # 畫出不同期限的預估線
                ax.axhline(y=ph1, color='red', ls=':', alpha=0.3, label="1D Res")
                ax.axhline(y=ph5, color='red', ls='--', alpha=0.6, label="5D Res")
                ax.axhline(y=ph10, color='red', ls='-', alpha=0.9, label="10D Res")
                ax.axhline(y=pl10, color='green', ls='-', alpha=0.9, label="10D Supp")
                
                ax.set_ylabel("Price (TWD)")
                ax.legend(loc='upper left', fontsize='small')
                st.pyplot(fig)
                st.caption("註：圖中實線為十日預估，虛線為五日預估，點狀線為隔日預估。")
            else:
                st.error("❌ 抓取不到數據")
