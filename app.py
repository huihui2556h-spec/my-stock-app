import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================================================
# 1. 系統初始化與導航
# =========================================================
st.set_page_config(page_title="台股 AI 多因子預測系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# 側邊欄導航選單
with st.sidebar:
    st.title("🎮 功能選單")
    if st.button("🏠 系統首頁", use_container_width=True): navigate_to("home")
    if st.button("⚡ 盤中即時預測", use_container_width=True): navigate_to("realtime")
    if st.button("📊 深度回測預判", use_container_width=True): navigate_to("forecast")
    st.divider()
    st.caption("狀態：FinMind & Volatility 引擎運作中")

# =========================================================
# 2. 核心計算函數
# =========================================================
def get_institutional_chips(stock_id):
    """計算籌碼修正因子 (FinMind)"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        chip_weight = 1.0 
        msg = "籌碼狀態：偏向中性"
        if not inst_df.empty:
            net = inst_df.tail(9)['buy'].sum() - inst_df.tail(9)['sell'].sum()
            if net > 0:
                chip_weight += 0.018
                msg = "✅ 籌碼強勢：法人近期大舉買超"
            else:
                chip_weight -= 0.018
                msg = "⚠️ 籌碼轉弱：法人近期持續調節"
        return round(chip_weight, 4), msg
    except:
        return 1.0, "⚠️ 籌碼資料讀取中..."

def ai_forecast_engine(df, chip_f=1.0):
    """AI 動態分位數預測引擎"""
    vol = df['Close'].pct_change().tail(20).std()
    h1_q, l1_q = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    h5_q, l5_q = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    
    df_c = df.tail(80).copy()
    df_c['h_pct'] = (df_c['High'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c['l_pct'] = (df_c['Low'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    
    return (df_c['h_pct'].quantile(h1_q) * chip_f, 
            df_c['l_pct'].quantile(l1_q) / chip_f,
            df_c['h_pct'].quantile(h5_q) * chip_f,
            df_c['l_pct'].quantile(l5_q) / chip_f)

def run_quad_backtest(df, chip_f):
    """計算四個獨立點位的命中率"""
    test_days = 20
    hist = df.tail(test_days + 65)
    hits = {"h1": 0, "l1": 0, "h5": 0, "l5": 0}
    for i in range(test_days):
        train = hist.iloc[i : i+60]
        pc = hist.iloc[i+60-1]['Close']
        h1, l1, h5, l5 = ai_forecast_engine(train, chip_f)
        if hist.iloc[i+60]['High'] >= pc * (1+h1): hits["h1"] += 1
        if hist.iloc[i+60]['Low'] <= pc * (1+l1): hits["l1"] += 1
        if hist.iloc[i+60:i+65]['High'].max() >= pc * (1+h5): hits["h5"] += 1
        if hist.iloc[i+60:i+65]['Low'].min() <= pc * (1+l5): hits["l5"] += 1
    return {k: (v/test_days)*100 for k, v in hits.items()}

# =========================================================
# 3. 頁面渲染邏輯 (含圖表中文詳細註解)
# =========================================================

# --- A. 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子動態回測系統")
    st.info("請利用左側導航選單進入『盤中即時預測』或『深度回測預判』。")

# --- B. 盤中即時預測 ---
elif st.session_state.mode == "realtime":
    st.title("⚡ 盤中即時點位監控")
    sid_rt = st.text_input("輸入股票代碼:")
    if sid_rt:
        df_rt = yf.download(f"{sid_rt}.TW", period="2d", interval="1m", progress=False)
        df_hist = yf.download(f"{sid_rt}.TW", period="200d", progress=False)
        if not df_rt.empty:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            now_price = float(df_rt['Close'].iloc[-1])
            chip_f, _ = get_institutional_chips(sid_rt)
            h1, l1, _, _ = ai_forecast_engine(df_hist, chip_f)
            acc = run_quad_backtest(df_hist, chip_f)

            st.metric("盤中即時成交價", f"{now_price:.2f}")
            col1, col2 = st.columns(2)
            col1.success(f"📈 即時壓力：{now_price*(1+h1):.2f} (準確率: {acc['h1']:.1f}%)")
            col2.error(f"📉 即時支撐：{now_price*(1+l1):.2f} (準確率: {acc['l1']:.1f}%)")
            
            # 即時圖表
            fig_rt, ax_rt = plt.subplots(figsize=(10, 3))
            ax_rt.plot(df_rt['Close'].tail(50), color="#1f77b4")
            ax_rt.axhline(now_price*(1+h1), color='red', ls='--')
            ax_rt.axhline(now_price*(1+l1), color='green', ls='--')
            st.pyplot(fig_rt)
            st.markdown("**📌 圖表中文註解：** 紅色虛線為 AI 預估今日壓力位，綠色虛線為支撐位。")

# --- C. 深度回測預判 ---
elif st.session_state.mode == "forecast":
    st.title("📊 深度回測與預判分析")
    sid_fc = st.text_input("輸入股票代碼 (例: 2330):")
    if sid_fc:
        df = yf.download(f"{sid_fc}.TW", period="200d", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            curr_c = float(df['Close'].iloc[-1])
            chip_f, chip_msg = get_institutional_chips(sid_fc)
            h1, l1, h5, l5 = ai_forecast_engine(df, chip_f)
            acc = run_quad_backtest(df, chip_f)

            st.metric("最新收盤基準價", f"{curr_c:.2f}")
            st.write(f"🧬 {chip_msg}")

            st.divider()
            # 數據盒子 (內含準確率)
            cA, cB = st.columns(2)
            with cA:
                st.info(f"📅 隔日壓力: {curr_c*(1+h1):.2f} | 準確率: {acc['h1']:.1f}%")
                st.info(f"🚩 五日壓力: {curr_c*(1+h5):.2f} | 準確率: {acc['h5']:.1f}%")
            with cB:
                st.success(f"📅 隔日支撐: {curr_c*(1+l1):.2f} | 準確率: {acc['l1']:.1f}%")
                st.success(f"⚓ 五日支撐: {curr_c*(1+l5):.2f} | 準確率: {acc['l5']:.1f}%")

            # 圖表
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Close'].tail(40), color="#1f77b4", label="Price")
            ax.axhline(curr_c*(1+h1), color='red', ls='--', label="T+1 High")
            ax.axhline(curr_c*(1+l1), color='green', ls='--', label="T+1 Low")
            ax.legend()
            st.pyplot(fig)

            # --- 重點：圖表中文註解區 ---
            st.markdown("""
            ### 📉 圖表中文註解說明
            1. **藍色實線 (Price)**：代表過去 40 個交易日的歷史收盤價走勢。
            2. **紅色虛線 (T+1 High)**：AI 根據**波動慣性**與**籌碼權重**計算出的**明日壓力預估線**。
            3. **綠色虛線 (T+1 Low)**：AI 計算出的**明日支撐預估線**。
            4. **灰白色帶狀區**：此區間為 AI 認定的「正常波動範圍」。若股價突破此範圍，代表發生超常態慣性。
            
            **💡 如何解讀準確率？**
            - **高準確率 (>75%)**：表示該股票近期走勢極其符合 AI 波動模型，點位參考價值極高。
            - **低準確率 (<40%)**：表示該股票近期可能處於強勢噴出或崩跌階段，傳統波動區間已被破壞，建議保守看待。
            """)
