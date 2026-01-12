import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta
import pytz

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 多因子動態回測系統", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🧬 外部籌碼資料庫：FinMind 分析 ---
def get_institutional_chips(stock_id):
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_dt)
        
        chip_weight = 1.0 
        analysis_note = ""
        
        if not inst_df.empty:
            recent_inst = inst_df.tail(9) 
            net_buy = recent_inst['buy'].sum() - recent_inst['sell'].sum()
            if net_buy > 0:
                chip_weight += 0.012
                analysis_note += "✅ 法人近三日合計買超。"
            elif net_buy < 0:
                chip_weight -= 0.012
                analysis_note += "⚠️ 法人近三日合計賣超。"
            
        if not margin_df.empty:
            m_recent = margin_df.tail(3)
            if m_recent['Margin_Purchase_today_balance'].iloc[-1] < m_recent['Margin_Purchase_today_balance'].iloc[0]:
                chip_weight += 0.005
                analysis_note += " ✅ 融資減少，籌碼趨穩。"
        
        return round(chip_weight, 4), analysis_note if analysis_note else "籌碼目前呈現中性震盪"
    except:
        return 1.0, "API 連線等待中 (請確認 requirements.txt 包含 FinMind)"

# --- 🧠 AI 動態優化引擎：尋找最佳波動參數 ---
def get_optimized_params(df):
    """根據個股近期的波動率 (Volatility) 自動調整分位數門檻"""
    recent_volatility = df['Close'].pct_change().tail(20).std()
    
    # 高波動股票 (如飆股)：需要更寬的預測帶 (更高的分位數)
    if recent_volatility > 0.025:
        h_q, l_q = 0.82, 0.18
    # 低波動股票 (如權值股)：需要較窄的預測帶 (較低的分位數)
    elif recent_volatility < 0.012:
        h_q, l_q = 0.68, 0.32
    else:
        h_q, l_q = 0.75, 0.25
    return h_q, l_q

def ai_dynamic_forecast_v3(df, chip_f=1.0):
    try:
        # 動態取得該股票專屬的優化參數
        h_q, l_q = get_optimized_params(df)
        
        df_clean = df.tail(60).copy()
        df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        
        # 結合「波動慣性」與「籌碼因子」進行動態位移
        h1_p = df_clean['h_pct'].quantile(h_q) * chip_f
        h5_p = df_clean['h_pct'].quantile(0.95) * chip_f
        l1_p = df_clean['l_pct'].quantile(l_q) / chip_f
        l5_p = df_clean['l_pct'].quantile(0.05) / chip_f
        
        return h1_p, h5_p, l1_p, l5_p
    except:
        return 0.02, 0.05, -0.015, -0.04

# --- 📊 實戰回測引擎：計算預測的可信度 ---
def backtest_engine(df, chip_f):
    """模擬過去 20 天，每天以當下的資料進行 AI 預測，計算真實命中率"""
    try:
        hits = 0
        test_days = 20
        # 準備資料：需要 test_days + 60 天的長度
        hist_data = df.tail(test_days + 60)
        
        for i in range(test_days):
            # 模擬歷史當天的視角 (只看得到當天以前的 60 天)
            train_window = hist_data.iloc[i : i+60]
            actual_high = hist_data.iloc[i+60]['High']
            actual_low = hist_data.iloc[i+60]['Low']
            prev_close = hist_data.iloc[i+60-1]['Close']
            
            # 使用當時的動態模型預測
            h1, _, l1, _ = ai_dynamic_forecast_v3(train_window, chip_f)
            pred_upper = prev_close * (1 + h1)
            pred_lower = prev_close * (1 + l1)
            
            # 判斷當日波動是否被 AI 區間「捕捉」到
            if actual_high >= pred_upper or actual_low <= pred_lower:
                hits += 1
        
        return (hits / test_days) * 100
    except:
        return 0.0

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
                動態修正值：{pct:.2f}%
            </span>
        </div>
    """, unsafe_allow_html=True)

# --- 🚀 頁面路由 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子動態回測系統")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時決策", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 深度回測預告", use_container_width=True): navigate_to("forecast")

# =========================================================
# ⚡ 盤中即時
# =========================================================
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    col_h, col_r = st.columns([4, 1.2])
    col_h.title("⚡ 盤中動態數據")
    if col_r.button("🔄 點擊重整", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    sid_rt = st.text_input("輸入台股代碼:", key="rt_v3")
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
        else: st.warning("目前無即時數據，請檢查代碼或是否為交易時間。")

# =========================================================
# 📊 深度回測預告 (動態回測版)
# =========================================================
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日 AI 多因子深度預判")
    sid_fc = st.text_input("輸入台股代碼:", key="fc_v3")
    if sid_fc:
        with st.spinner('AI 正進行 20 日滾動回測與籌碼因子校正...'):
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
                
                # 2. AI 動態預測 (最高低位)
                h1, h5, l1, l5 = ai_dynamic_forecast_v3(df, chip_f=chip_f)
                ph1, ph5 = curr_c*(1+h1), curr_c*(1+h5)
                pl1, pl5 = curr_c*(1+l1), curr_c*(1+l5)

                # 3. 核心：計算 20 日實戰回測準確率
                bt_accuracy = backtest_engine(df, chip_f)

                st.subheader(f"🏠 {get_stock_name(sid_fc)}")
                
                # 顯示準確率標籤
                acc_color = "green" if bt_accuracy >= 70 else "orange"
                st.markdown(f"### AI 實戰回測準確率：<span style='color:{acc_color}'>{bt_accuracy:.1f}%</span>", unsafe_allow_html=True)
                st.caption("*(註：此準確率是模擬過去 20 個交易日「每日早盤預測」的真實命中結果)*")

                with st.expander("🧬 AI 信心指數與籌碼說明", expanded=True):
                    st.write(f"**信心權重：{chip_f:.3f}**")
                    st.info(f"**籌碼狀態：** {chip_msg}")

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    render_box("📈 隔日最高動態預估", ph1, h1*100, "red")
                    render_box("🚩 五日極限高點預估", ph5, h5*100, "red")
                with col2:
                    render_box("📉 隔日最低動態預估", pl1, l1*100, "green")
                    render_box("⚓ 五日極限低點預估", pl5, l5*100, "green")

                # 具體買賣點位
                st.divider()
                st.markdown("### 🏹 隔日買賣實戰建議")
                d1, d2, d3 = st.columns(3)
                # 基於動態回測點位的 40% 作為極其保守進場，80% 為停利
                buy_in = curr_c * (1 + (l1 * 0.4)) 
                target_win = curr_c * (1 + (h1 * 0.8))
                d1.info(f"🔹 **建議進場區間**\n\n{buy_in:.2f} ~ {curr_c:.2f}")
                d2.error(f"🔹 **關鍵防守參考**\n\n{pl1:.2f}")
                d3.success(f"🔸 **AI 預估停利位**\n\n{target_win:.2f}")

                # 圖表
                st.divider()
                st.write("### 📉 波動慣性與 AI 預測區間")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
                ax1.plot(df.index[-40:], df['Close'].tail(40), color='#1f77b4', lw=2, label="Price")
                ax1.axhline(y=ph1, color='red', ls='--', alpha=0.5, label="Dynamic Resistance")
                ax1.axhline(y=pl1, color='green', ls='--', alpha=0.5, label="Dynamic Support")
                ax1.fill_between(df.index[-40:], pl1, ph1, color='gray', alpha=0.1, label="AI Prediction Zone")
                ax1.legend()
                
                pdf = df.tail(40); clrs = ['red' if pdf['Close'].iloc[i] >= pdf['Open'].iloc[i] else 'green' for i in range(len(pdf))]
                ax2.bar(pdf.index, pdf['Volume'], color=clrs, alpha=0.6)
                st.pyplot(fig)
                
                st.markdown(f"""
                #### 📝 動態分析說明
                1. **自適應模型**：AI 偵測到此股近期波動率為 `{df['Close'].pct_change().tail(20).std():.4f}`，已自動調整預測分位數。
                2. **回測機制**：畫面頂部的 `{bt_accuracy:.1f}%` 是透過 **Walk-forward (滾動式驗證)** 計算，比傳統靜態達成率更具實戰意義。
                3. **籌碼聯動**：預估區間已隨法人買賣力道 `{chip_f}` 進行位移修正。
                """)
