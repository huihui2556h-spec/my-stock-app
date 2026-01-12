import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

# =========================================================
# 1. 系統初始化與頁面導航
# =========================================================
st.set_page_config(page_title="台股 AI 深度預測與當沖決策系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 判斷交易時段 ---
def is_market_open():
    now = datetime.now()
    if now.weekday() > 4: return False
    return time(9, 0) <= now.time() <= time(13, 30)

# =========================================================
# 2. 核心數據引擎 (AI 預測與深度命中率)
# =========================================================
def get_advanced_data(stock_id):
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start = (datetime.now() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        weight = 1.0
        msg = "籌碼中性"
        if not df_inst.empty:
            net = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net > 0: weight += 0.02; msg = "✅ 籌碼強勢：法人買盤主導"
            else: weight -= 0.02; msg = "⚠️ 籌碼轉弱：法人調節中"
        return weight, msg
    except:
        return 1.0, "⚠️ 外部數據同步中"

def ai_deep_engine(df, chip_f=1.0):
    """計算隔日、五日最高價與各自命中率"""
    # 波動率計算
    vol = df['Close'].pct_change().tail(20).std()
    
    # 歷史百分比序列
    df_c = df.tail(100).copy()
    df_c['h_pct'] = (df_c['High'] - df_c['Close'].shift(1)) / df_c['Close'].shift(1)
    
    # 動態分位數 (由波動慣性決定)
    # 隔日最高價分位數 (通常取 0.7-0.9 之間)
    q_h1 = 0.85 if vol > 0.02 else 0.75
    # 五日最高價分位數 (取 0.92-0.97 之間)
    q_h5 = 0.95 if vol > 0.02 else 0.92
    
    h1_pct = df_c['h_pct'].quantile(q_h1) * chip_f
    h5_pct = df_c['h_pct'].quantile(q_h5) * chip_f
    
    # 實戰命中率回測 (過去 20 天)
    test_days = 20
    hist = df.tail(80)
    hits = {"h1": 0, "h5": 0}
    for i in range(test_days):
        train = hist.iloc[i : i+60]
        pc = hist.iloc[i+60-1]['Close']
        # 模擬當時的預估
        pred_h1 = train['h_pct'].quantile(q_h1) * chip_f
        pred_h5 = train['h_pct'].quantile(q_h5) * chip_f
        # 檢查是否觸及
        if hist.iloc[i+60]['High'] >= pc * (1 + pred_h1): hits["h1"] += 1
        if hist.iloc[i+60:i+65]['High'].max() >= pc * (1 + pred_h5): hits["h5"] += 1
        
    return h1_pct, h5_pct, (hits["h1"]/test_days)*100, (hits["h5"]/test_days)*100

# =========================================================
# 3. 頁面邏輯內容
# =========================================================

# --- A. 首頁 (雙按鈕直覺導向) ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 深度預測與當沖決策系統")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.info("### ⚡ 盤中即時預測")
        st.write("交易時間即時監控最高點目標。")
        if st.button("進入盤中監控", use_container_width=True): navigate_to("realtime")
    with c2:
        st.success("### 📊 深度預估分析")
        st.write("預測隔日與五日最高價、成交量分析與當沖建議。")
        if st.button("進入深度預判", use_container_width=True): navigate_to("forecast")

# --- B. 盤中即時預測 ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時最高點監控")
    rt_sid = st.text_input("輸入股票代碼 (例: 2330):", key="rt_sid")
    if rt_sid:
        if not is_market_open():
            st.warning("🏮 目前尚未開盤。開放時間：週一至週五 09:00 - 13:30")
        else:
            with st.spinner("抓取即時數據中..."):
                df_rt = yf.download(f"{rt_sid}.TW", period="1d", interval="1m", progress=False)
                df_h = yf.download(f"{rt_sid}.TW", period="150d", progress=False)
                if not df_rt.empty:
                    if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
                    now_p = float(df_rt['Close'].iloc[-1])
                    chip_w, _ = get_advanced_data(rt_sid)
                    h1_p, _, acc1, _ = ai_deep_engine(df_h, chip_w)
                    st.metric(f"🚀 {rt_sid} 盤中現價", f"{now_p:.2f}")
                    st.error(f"🎯 今日預估最高點：{now_p*(1+h1_p):.2f} (達成機率: {acc1:.1f}%)")

# --- C. 深度回測預判 (修復最高價與彩色價量) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日與五日最高價預判及當沖建議")
    fc_sid = st.text_input("請輸入分析代碼 (例: 2603):", key="fc_sid")
    
    if fc_sid:
        with st.spinner("深度數據計算中..."):
            df = yf.download(f"{fc_sid}.TW", period="250d", progress=False)
            if df.empty: df = yf.download(f"{fc_sid}.TWO", period="250d", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                curr_c = float(df['Close'].iloc[-1])
                chip_w, chip_msg = get_advanced_data(fc_sid)
                h1, h5, acc1, acc5 = ai_deep_engine(df, chip_w)

                st.subheader(f"🏠 分析報告：{fc_sid}")
                st.metric("📌 最新收盤基準價", f"{curr_c:.2f}")
                st.write(f"🧬 {chip_msg}")

                st.divider()
                # --- 強調最高價與達成率 ---
                st.markdown("### 🎯 AI 預估最高價位目標")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div style="background:#fff5f5; border:1px solid #ffcccc; padding:20px; border-radius:10px;">
                        <h4 style="color:#e63946; margin:0;">📈 隔日預估最高價</h4>
                        <h1 style="color:#111; margin:10px 0;">{(curr_c*(1+h1)):.2f}</h1>
                        <p style="color:#555; font-size:14px;">預估漲幅: {h1*100:.2f}% | 🎯 歷史達成率: <b>{acc1:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div style="background:#f0f7ff; border:1px solid #cce3ff; padding:20px; border-radius:10px;">
                        <h4 style="color:#0077b6; margin:0;">🚩 五日預估最高價</h4>
                        <h1 style="color:#111; margin:10px 0;">{(curr_c*(1+h5)):.2f}</h1>
                        <p style="color:#555; font-size:14px;">預估漲幅: {h5*100:.2f}% | 🎯 歷史達成率: <b>{acc5:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                # --- 當沖決策建議 ---
                st.markdown("### ⚡ 當沖/隔日沖實戰操作建議")
                s1, s2, s3 = st.columns(3)
                s1.info(f"⚓ 建議進場位\n\n**{curr_c * 1.002:.2f}**\n(開盤平盤上站穩)")
                s2.error(f"🚀 當沖停利目標\n\n**{curr_c*(1+h1*0.96):.2f}**\n(預估最高價前退場)")
                s3.warning(f"🛑 停損防守位\n\n**{curr_c * 0.985:.2f}**\n(跌破 1.5% 需止損)")

                # --- 彩色價量表 ---
                st.divider()
                st.write("### 📈 彩色價量趨勢分析 (Color-Coded Volume)")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                plot_df = df.tail(40).copy()
                # 價格線
                ax1.plot(plot_df.index, plot_df['Close'], color="#1f77b4", lw=2, label="Close Price")
                ax1.axhline(curr_c*(1+h1), color='red', ls='--', label="T+1 High Target")
                ax1.set_title("Price and Predicted High")
                ax1.legend()

                # 彩色成交量：漲紅跌綠 (台股慣例)
                colors = ['#e63946' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else '#2a9d8f' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.8)
                ax2.set_title("Volume (Red=Up, Green=Down)")
                
                plt.xticks(rotation=45)
                st.pyplot(fig)

                st.markdown("""
                **📊 數據深度解讀：**
                1. **最高價邏輯**：隔日預估最高價是結合過去波動幅度的分位數與籌碼權重算出，**達成率**代表過去 20 天中有多少天實質觸及此價位。
                2. **彩色成交量**：**紅色柱狀**代表收紅盤（買盤強），**綠色柱狀**代表收黑盤（賣盤強）。若股價接近預估最高價且量能爆出紅柱，代表多頭動能極強。
                3. **五日最高價**：此為波段觀察點，若五日達成率低於 50%，代表該點位壓力極重，不建議過度追高。
                """)
