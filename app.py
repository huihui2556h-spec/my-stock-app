import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta

# 1. 頁面配置與導航初始化
st.set_page_config(page_title="台股 AI 多因子動態回測系統 Pro", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🧬 外部籌碼資料庫：FinMind 分析 ---
def get_institutional_chips(stock_id):
    """
    抓取法人籌碼並計算動態權重因子。
    """
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 考慮週末與連假，抓取近 14 天資料
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_dt)
        
        chip_weight = 1.0 
        analysis_note = ""
        
        if not inst_df.empty:
            recent_inst = inst_df.tail(9) # 三大法人各三筆
            net_buy = recent_inst['buy'].sum() - recent_inst['sell'].sum()
            if net_buy > 0:
                chip_weight += 0.012
                analysis_note += "✅ 法人近三日買超；"
            elif net_buy < 0:
                chip_weight -= 0.012
                analysis_note += "⚠️ 法人近三日賣超；"
            
        if not margin_df.empty:
            m_recent = margin_df.tail(3)
            # 融資餘額減少代表籌碼流向大戶，有利穩定
            if m_recent['Margin_Purchase_today_balance'].iloc[-1] < m_recent['Margin_Purchase_today_balance'].iloc[0]:
                chip_weight += 0.005
                analysis_note += "✅ 融資餘額減少(籌碼趨穩)。"
            else:
                analysis_note += "❌ 融資餘額增加。"
        
        return round(chip_weight, 4), analysis_note if analysis_note else "籌碼目前呈中性"
    except:
        return 1.0, "FinMind API 連線中或今日資料尚未更新"

# --- 🧠 AI 動態預測核心 V4 (自適應波動優化) ---
def ai_dynamic_forecast_v4(df, chip_f=1.0):
    """
    核心邏輯：不再使用固定分位數，根據近期標準差動態調整門檻。
    """
    try:
        # 計算近 20 日波動率 (Standard Deviation)
        vol = df['Close'].pct_change().tail(20).std()
        
        # 動態分位數：波動大時拉寬門檻 (0.82)，波動小時縮小門檻 (0.72)
        h_q = 0.82 if vol > 0.02 else 0.72
        l_q = 0.18 if vol > 0.02 else 0.28
        
        df_clean = df.tail(60).copy()
        df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        
        h1_p = df_clean['h_pct'].quantile(h_q) * chip_f
        h5_p = df_clean['h_pct'].quantile(0.95) * chip_f
        l1_p = df_clean['l_pct'].quantile(l_q) / chip_f
        l5_p = df_clean['l_pct'].quantile(0.05) / chip_f
        
        return h1_p, h5_p, l1_p, l5_p
    except:
        return 0.02, 0.05, -0.015, -0.04

# --- 📊 活化回測：計算「動態準確率曲線」 ---
def get_backtest_series(df, chip_f):
    """
    模擬過去 30 天實戰，回傳每日命中狀態與滾動準確率。
    """
    try:
        test_days = 30
        hist_data = df.tail(test_days + 60)
        hit_series = []
        dates = []
        
        for i in range(test_days):
            train_window = hist_data.iloc[i : i+60]
            actual_h = hist_data.iloc[i+60]['High']
            actual_l = hist_data.iloc[i+60]['Low']
            prev_c = hist_data.iloc[i+60-1]['Close']
            date = hist_data.index[i+60]
            
            h1, _, l1, _ = ai_dynamic_forecast_v4(train_window, chip_f)
            # 命中定義：股價當天有達到 AI 預測的高點或低點區域
            is_hit = 1 if (actual_h >= prev_c*(1+h1) or actual_l <= prev_c*(1+l1)) else 0
            hit_series.append(is_hit)
            dates.append(date)
            
        # 計算 5 日滾動平均命中率
        acc_rolling = pd.Series(hit_series).rolling(window=5, min_periods=1).mean() * 100
        return dates, acc_rolling.tolist()
    except:
        return [], []

def get_stock_name(sid):
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"股票 {sid}"

def render_box(label, price, pct, color="red"):
    c_code = "#FF4B4B" if color == "red" else "#28A745"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {c_code}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{c_code}; color:white; padding:2px 8px; border-radius:5px; font-size:13px;">
                預估振幅：{pct:.2f}%
            </span>
        </div>
    """, unsafe_allow_html=True)

# --- 🚀 頁面路由與邏輯 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子動態回測系統 Pro")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時數據", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 深度回測預告", use_container_width=True): navigate_to("forecast")

# =========================================================
# ⚡ 盤中即時模式
# =========================================================
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中動態數據決策")
    sid_rt = st.text_input("輸入台股代碼 (如 2330):", key="rt_input")
    if sid_rt:
        success = False
        for suf in [".TW", ".TWO"]:
            df_rt = yf.download(f"{sid_rt}{suf}", period="1d", interval="1m", progress=False)
            if not df_rt.empty:
                success = True; break
        if success:
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            df_rt['VWAP'] = (df_rt['Close'] * df_rt['Volume']).cumsum() / df_rt['Volume'].cumsum()
            cp = float(df_rt['Close'].iloc[-1])
            vp = float(df_rt['VWAP'].iloc[-1])
            st.subheader(f"🎯 {get_stock_name(sid_rt)}")
            st.metric("即時現價", f"{cp:.2f}")
            st.success(f"🔹 即時動態支撐 (VWAP)：{vp:.2f}")
            st.error(f"🔸 建議短線獲利位：{cp * 1.015:.2f}")
        else: st.warning("目前無即時數據。")

# =========================================================
# 📊 深度回測預告模式 (動態達成率版)
# =========================================================
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日 AI 多因子深度預判")
    sid_fc = st.text_input("輸入台股代碼 (如 2603):", key="fc_input")
    
    if sid_fc:
        with st.spinner('AI 正進行 30 日逐日滾動回測與籌碼校正...'):
            success = False
            for suf in [".TW", ".TWO"]:
                df = yf.download(f"{sid_fc}{suf}", period="150d", progress=False)
                if not df.empty:
                    success = True; break
            
            if success:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                curr_c = float(df['Close'].iloc[-1])
                
                # 1. 執行籌碼分析
                chip_f, chip_msg = get_institutional_chips(sid_fc)
                
                # 2. 獲取動態回測曲線 (活化達成率)
                dates, acc_curve = get_backtest_series(df, chip_f)
                current_acc = acc_curve[-1] if acc_curve else 0
                
                # 3. AI 隔日點位預測
                h1, h5, l1, l5 = ai_dynamic_forecast_v4(df, chip_f=chip_f)
                ph1, ph5 = curr_c*(1+h1), curr_c*(1+h5)
                pl1, pl5 = curr_c*(1+l1), curr_c*(1+l5)

                # --- UI 顯示區 ---
                st.subheader(f"🏠 {get_stock_name(sid_fc)} 實戰分析報告")
                
                # 活化的達成率：以滾動圖表呈現
                acc_color = "green" if current_acc >= 75 else "orange"
                st.markdown(f"### 當前 AI 預測可靠度：<span style='color:{acc_color}'>{current_acc:.1f}%</span>", unsafe_allow_html=True)
                
                fig_acc, ax_acc = plt.subplots(figsize=(10, 2.5))
                ax_acc.plot(dates, acc_curve, color='#FF4B4B', lw=2, label="AI Rolling Accuracy (5D)")
                ax_acc.fill_between(dates, acc_curve, color='#FF4B4B', alpha=0.1)
                ax_acc.set_ylim(0, 105)
                ax_acc.set_title("歷史達成率趨勢 (活化診斷)")
                st.pyplot(fig_acc)
                st.caption("▲ 這條曲線若處於高點，代表近期股價慣性符合 AI 邏輯；若曲線暴跌，代表目前市場處於變盤期，應降低槓桿。")

                with st.expander("🧬 籌碼信心指數解析", expanded=True):
                    st.write(f"**當前修正因子：{chip_f}**")
                    st.info(f"**實時狀態：** {chip_msg}")

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    render_box("📈 隔日最高動態預估", ph1, h1*100, "red")
                    render_box("🚩 五日極限高點預估", ph5, h5*100, "red")
                with col2:
                    render_box("📉 隔日最低動態預估", pl1, l1*100, "green")
                    render_box("⚓ 五日極限低點預估", pl5, l5*100, "green")

                # 實戰點位建議
                st.divider()
                st.markdown("### 🏹 隔日實戰買賣點位")
                d1, d2, d3 = st.columns(3)
                buy_in = curr_c * (1 + (l1 * 0.4)) 
                target_win = curr_c * (1 + (h1 * 0.8))
                d1.info(f"🔹 **建議進場區間**\n\n{buy_in:.2f} ~ {curr_c:.2f}")
                d2.error(f"🔹 **關鍵防守參考**\n\n{pl1:.2f}")
                d3.success(f"🔸 **AI 預估停利位**\n\n{target_win:.2f}")

                # 走勢圖與說明
                st.divider()
                st.write("### 📉 波動慣性與 AI 預測帶")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
                ax1.plot(df.index[-40:], df['Close'].tail(40), color='#1f77b4', lw=2, label="Price")
                ax1.axhline(y=ph1, color='red', ls='--', alpha=0.5, label="AI Resistance")
                ax1.axhline(y=pl1, color='green', ls='--', alpha=0.5, label="AI Support")
                ax1.fill_between(df.index[-40:], pl1, ph1, color='gray', alpha=0.1)
                ax1.legend()
                pdf = df.tail(40); clrs = ['red' if pdf['Close'].iloc[i] >= pdf['Open'].iloc[i] else 'green' for i in range(len(pdf))]
                ax2.bar(pdf.index, pdf['Volume'], color=clrs, alpha=0.6)
                st.pyplot(fig)
                
                st.markdown(f"""
                #### 📝 圖表文字說明
                1. **動態達成率圖**：顯示過去 30 天 AI 捕捉波動的成功率。**波峰**代表 AI 節奏正確，**波谷**代表市場目前不按牌理出牌。
                2. **預估振幅**：此數值非固定，而是根據最近 20 天的 `{df['Close'].pct_change().tail(20).std():.4f}` 標準差自動適應。
                3. **點位策略**：建議進場點參考「隔日最低預估」的緩衝位，停利則設定在預期高點的 80% 處以求穩健。
                """)
