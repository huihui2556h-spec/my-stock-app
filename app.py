import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================
# 1. 系統初始化與導航邏輯
# =========================================================
st.set_page_config(page_title="台股 AI 多因子動態回測系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# 側邊欄：永遠存在的逃生艙
with st.sidebar:
    st.title("⚙️ 系統選單")
    if st.button("🏠 回到首頁", use_container_width=True):
        navigate_to("home")
    st.divider()
    st.caption("版本：v2.6 (FinMind 整合版)")

# =========================================================
# 2. 判斷是否為盤中時間
# =========================================================
def is_market_open():
    now = datetime.now()
    # 判斷週一到週五
    if now.weekday() > 4:
        return False
    current_time = now.time()
    # 09:00 - 13:30
    start_time = time(9, 0)
    end_time = time(13, 30)
    return start_time <= current_time <= end_time

# =========================================================
# 3. 核心運算 (AI 與 籌碼)
# =========================================================
def get_chips(stock_id):
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start = (datetime.now() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        weight = 1.0
        msg = "籌碼中性"
        if not df_inst.empty:
            net = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net > 0: weight += 0.015; msg = "✅ 法人連日買超"
            else: weight -= 0.015; msg = "⚠️ 法人連日調節"
        return weight, msg
    except:
        return 1.0, "⚠️ 籌碼 API 連線中"

def ai_engine(df, chip_f=1.0):
    vol = df['Close'].pct_change().tail(20).std()
    h1_q, l1_q = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    h5_q, l5_q = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    df_c = df.tail(80).copy()
    df_c['h_pct'] = (df_c['High'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    df_c['l_pct'] = (df_c['Low'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    return (df_c['h_pct'].quantile(h1_q)*chip_f, df_c['l_pct'].quantile(l1_q)/chip_f,
            df_c['h_pct'].quantile(h5_q)*chip_f, df_c['l_pct'].quantile(l5_q)/chip_f)

def run_backtest(df, chip_f):
    test_days = 20
    hist = df.tail(85)
    hits = {"h1":0, "l1":0, "h5":0, "l5":0}
    for i in range(test_days):
        train = hist.iloc[i : i+60]
        pc = hist.iloc[i+60-1]['Close']
        h1, l1, h5, l5 = ai_engine(train, chip_f)
        if hist.iloc[i+60]['High'] >= pc*(1+h1): hits["h1"]+=1
        if hist.iloc[i+60]['Low'] <= pc*(1+l1): hits["l1"]+=1
        if hist.iloc[i+60:i+65]['High'].max() >= pc*(1+h5): hits["h5"]+=1
        if hist.iloc[i+60:i+65]['Low'].min() <= pc*(1+l5): hits["l5"]+=1
    return {k: (v/test_days)*100 for k, v in hits.items()}

# =========================================================
# 4. 頁面邏輯
# =========================================================

# --- A. 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子動態回測系統 Pro")
    st.write("請選擇您要使用的分析模式：")
    col1, col2 = st.columns(2)
    with col1:
        st.info("### ⚡ 盤中即時預測")
        st.write("監控盤中即時價格，對照當日 AI 壓力與支撐。")
        if st.button("點此進入盤中模式", use_container_width=True): navigate_to("realtime")
    with col2:
        st.success("### 📊 隔日深度回測")
        st.write("根據收盤數據預測隔日與五日目標，並查看命中率。")
        if st.button("點此進入回測模式", use_container_width=True): navigate_to("forecast")

# --- B. 盤中即時預測 ---
elif st.session_state.mode == "realtime":
    st.title("⚡ 盤中即時點位監控")
    rt_sid = st.text_input("輸入股票代碼 (例: 2330):", key="rt_sid")
    if rt_sid:
        if not is_market_open():
            st.warning("🏮 目前尚未開盤。盤中即時預測僅在週一至週五 09:00 - 13:30 開放。")
        else:
            with st.spinner("正在獲取即時成交資訊..."):
                df_rt = yf.download(f"{rt_sid}.TW", period="1d", interval="1m", progress=False)
                df_hist = yf.download(f"{rt_sid}.TW", period="200d", progress=False)
                if not df_rt.empty:
                    if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
                    now_p = float(df_rt['Close'].iloc[-1])
                    chip_w, chip_m = get_chips(rt_sid)
                    h1, l1, _, _ = ai_engine(df_hist, chip_w)
                    acc = run_backtest(df_hist, chip_w)

                    st.metric(f"🚀 {rt_sid} 盤中現價", f"{now_p:.2f}")
                    st.write(f"🧬 {chip_m}")
                    
                    c1, c2 = st.columns(2)
                    c1.error(f"當日預估壓力: {now_p*(1+h1):.2f} (準確率: {acc['h1']:.1f}%)")
                    c2.success(f"當日預估支撐: {now_p*(1+l1):.2f} (準確率: {acc['l1']:.1f}%)")
                    
                    fig_rt, ax_rt = plt.subplots(figsize=(10, 3))
                    ax_rt.plot(df_rt['Close'], color="#1f77b4")
                    ax_rt.axhline(now_p*(1+h1), color='red', ls='--')
                    ax_rt.axhline(now_p*(1+l1), color='green', ls='--')
                    st.pyplot(fig_rt)
                    st.caption("圖表註解：紅色虛線為當日預期壓力，綠色虛線為當日預期支撐。")

# --- C. 隔日深度回測 (修正失效問題) ---
elif st.session_state.mode == "forecast":
    st.title("📊 隔日與五日深度預判分析")
    # 確保 key 唯一，且邏輯完整觸發
    fc_sid = st.text_input("請輸入要分析的代碼 (例如: 2603):", key="fc_sid_unique")
    if fc_sid:
        with st.spinner(f"正在計算 {fc_sid} 的波動慣性與回測數據..."):
            df = yf.download(f"{fc_sid}.TW", period="200d", progress=False)
            if df.empty: df = yf.download(f"{fc_sid}.TWO", period="200d", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                curr_c = float(df['Close'].iloc[-1])
                chip_w, chip_m = get_chips(fc_sid)
                h1, l1, h5, l5 = ai_engine(df, chip_w)
                bt = run_backtest(df, chip_w)

                st.subheader(f"🏠 分析報告：{fc_sid}")
                st.metric("📌 最新收盤基準價", f"{curr_c:.2f}")
                st.info(f"🧬 {chip_m}")

                st.divider()
                st.markdown("### 🎯 預估點位與各自命中率")
                cA, cB = st.columns(2)
                with cA:
                    st.error(f"📅 隔日壓力 (T+1): {curr_c*(1+h1):.2f} | 🎯 準確率: {bt['h1']:.1f}%")
                    st.error(f"🚩 五日壓力 (T+5): {curr_c*(1+h5):.2f} | 🎯 準確率: {bt['h5']:.1f}%")
                with cB:
                    st.success(f"📅 隔日支撐 (T+1): {curr_c*(1+l1):.2f} | 🎯 準確率: {bt['l1']:.1f}%")
                    st.success(f"⚓ 五日支撐 (T+5): {curr_c*(1+l5):.2f} | 🎯 準確率: {bt['l5']:.1f}%")

                # 圖表顯示
                fig, ax = plt.subplots(figsize=(10, 4))
                hist_p = df['Close'].tail(40)
                ax.plot(hist_p.index, hist_p, label="Price", color="#1f77b4")
                ax.axhline(curr_c*(1+h1), color='red', ls='--', label="T+1 High")
                ax.axhline(curr_c*(1+l1), color='green', ls='--', label="T+1 Low")
                ax.legend()
                st.pyplot(fig)

                st.markdown("""
                ### 📉 圖表中文註解說明
                1. **紅/綠虛線**：分別代表 AI 預估的隔日壓力位與支撐位。
                2. **命中率解讀**：若隔日支撐的命中率遠高於壓力，表示近期股價偏向回檔測底；反之則慣性向上。
                3. **五日預判**：適合週轉期較長的交易者，衡量一週內的波段空間。
                """)
            else:
                st.error("查無數據，請確認代碼是否輸入正確（如 2330）。")
