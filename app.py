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
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 🧬 外部籌碼資料庫：FinMind 強化分析 ---
def get_institutional_chips(stock_id):
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取近 14 天以確保數據涵蓋最新交易日
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_dt)
        
        chip_weight = 1.0 
        analysis_note = ""
        
        if not inst_df.empty:
            recent_inst = inst_df.tail(9) 
            net_buy = recent_inst['buy'].sum() - recent_inst['sell'].sum()
            # 依據法人買賣力道進行權重修正
            if net_buy > 0:
                chip_weight += 0.012
                analysis_note += "✅ 法人近三日合計買超。"
            elif net_buy < 0:
                chip_weight -= 0.012
                analysis_note += "⚠️ 法人近三日合計賣超。"
            
        if not margin_df.empty:
            m_recent = margin_df.tail(3)
            # 融資減少通常代表籌碼由散戶流向大戶，有利漲勢
            if m_recent['Margin_Purchase_today_balance'].iloc[-1] < m_recent['Margin_Purchase_today_balance'].iloc[0]:
                chip_weight += 0.005
                analysis_note += " ✅ 融資餘額減少，籌碼穩定。"
            else:
                analysis_note += " ❌ 融資餘額增加，籌碼趨於分散。"
        
        return round(chip_weight, 4), analysis_note if analysis_note else "數據更新中或維持中性"
    except:
        return 1.0, "API 連線異常或今日數據尚未公告 (預計 16:30 更新)"

# --- 🧠 AI 動態預測核心 ---
def ai_dynamic_forecast(df, chip_f=1.0):
    try:
        df_clean = df.tail(60).copy()
        df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
        
        # 核心 AI 邏輯：歷史波動分位數 * 籌碼因子
        h1_p = df_clean['h_pct'].quantile(0.75) * chip_f
        h5_p = df_clean['h_pct'].quantile(0.95) * chip_f
        l1_p = df_clean['l_pct'].quantile(0.25) / chip_f
        l5_p = df_clean['l_pct'].quantile(0.05) / chip_f
        
        return h1_p, h5_p, l1_p, l5_p
    except:
        return 0.02, 0.05, -0.015, -0.04

def calculate_real_accuracy(df, target_p, side='high'):
    try:
        df_c = df.copy().tail(60)
        hits = 0
        for i in range(1, len(df_c)):
            prev_c = df_c['Close'].iloc[i-1]
            actual = df_c['High'].iloc[i] if side == 'high' else df_c['Low'].iloc[i]
            pred = prev_c * (1 + target_p)
            if side == 'high' and actual >= pred: hits += 1
            elif side == 'low' and actual <= pred: hits += 1
        return (hits / len(df_c)) * 100
    except: return 0.0

def get_stock_name(sid):
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"股票 {sid}"

def render_box(label, price, pct, acc, color="red"):
    c_code = "#FF4B4B" if color == "red" else "#28A745"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {c_code}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{c_code}; color:white; padding:2px 8px; border-radius:5px; font-size:13px;">
                AI 預估振幅：{pct:.2f}%
            </span>
            <p style="margin-top:10px; font-size:11px; color:#888;">↳ 歷史特徵達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# --- 🚀 頁面路由 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ 盤中即時量價", use_container_width=True): navigate_to("realtime")
    with c2:
        if st.button("📊 隔日深度預估", use_container_width=True): navigate_to("forecast")

# =========================================================
# ⚡ 盤中即時
# =========================================================
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    col_h, col_r = st.columns([4, 1.2])
    col_h.title("⚡ 盤中動態決策")
    if col_r.button("🔄 點擊重整", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    sid_rt = st.text_input("輸入台股代碼:", key="rt_input")
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
            st.success(f"🔹 即時支撐 (VWAP)：{vp:.2f}")
            st.error(f"🔸 即時建議停利：{cp * 1.015:.2f}")
        else: st.warning("目前無即時成交數據。")

# =========================================================
# 📊 隔日深度預估 (整合信心解釋、買賣建議、圖表說明)
# =========================================================
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日多因子 AI 預判")
    sid_fc = st.text_input("輸入台股代碼:", key="fc_input")
    if sid_fc:
        with st.spinner('正在分析波動慣性、法人籌碼並產生預測...'):
            success = False
            for suf in [".TW", ".TWO"]:
                df = yf.download(f"{sid_fc}{suf}", period="100d", progress=False)
                if not df.empty:
                    success = True; break
            
            if success:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                curr_c = float(df['Close'].iloc[-1])
                
                # 1. 獲取法人籌碼與詳細解釋
                chip_f, chip_msg = get_institutional_chips(sid_fc)
                
                # 2. AI 動態預測
                h1, h5, l1, l5 = ai_dynamic_forecast(df, chip_f=chip_f)
                ph1, ph5 = curr_c*(1+h1), curr_c*(1+h5)
                pl1, pl5 = curr_c*(1+l1), curr_c*(1+l5)

                st.subheader(f"🏠 {get_stock_name(sid_fc)}")
                st.metric("今日收盤價", f"{curr_c:.2f}")
                
                # --- 信心指數文字敘述 ---
                with st.expander("🧬 AI 信心指數說明", expanded=True):
                    st.write(f"**目前數值：{chip_f:.3f}**")
                    st.info(f"**圖表解讀建議：** {chip_msg}")
                    st.caption("此數值若大於 1.05 代表籌碼極度集中；小於 0.95 代表法人持續撤出。AI 會根據此數值動態修正預測點位的高低。")

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    render_box("📈 隔日最高預估", ph1, h1*100, calculate_real_accuracy(df, h1, 'high'), "red")
                    render_box("🚩 五日最高預估", ph5, h5*100, calculate_real_accuracy(df, h5, 'high'), "red")
                with col2:
                    render_box("📉 隔日最低預估", pl1, l1*100, calculate_real_accuracy(df, l1, 'low'), "green")
                    render_box("⚓ 五日最低預估", pl5, l5*100, calculate_real_accuracy(df, l5, 'low'), "green")

                # --- 隔日買賣價格具體建議 ---
                st.divider()
                st.markdown("### 🏹 隔日實戰買賣建議點位")
                d1, d2, d3 = st.columns(3)
                # 基於 AI 低位預判的 50% 振幅作為保守進場點
                buy_in = curr_c * (1 + (l1 * 0.5)) 
                # 基於 AI 高位預判的 85% 作為保守停利點
                target_win = curr_c * (1 + (h1 * 0.85))

                d1.info(f"🔹 **多方建議進場位**\n\n{buy_in:.2f}")
                d2.error(f"🔹 **空方/防守參考位**\n\n{pl1:.2f}")
                d3.success(f"🔸 **隔日獲利目標位**\n\n{target_win:.2f}")

                # 圖表展示
                st.divider()
                st.write("### 📉 歷史走勢與 AI 預測帶")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
                ax1.plot(df.index[-40:], df['Close'].tail(40), color='#1f77b4', lw=2, label="Price")
                ax1.axhline(y=ph1, color='red', ls='--', alpha=0.4, label="AI Resistance")
                ax1.axhline(y=pl1, color='green', ls='--', alpha=0.4, label="AI Support")
                ax1.legend()
                
                pdf = df.tail(40)
                clrs = ['red' if pdf['Close'].iloc[i] >= pdf['Open'].iloc[i] else 'green' for i in range(len(pdf))]
                ax2.bar(pdf.index, pdf['Volume'], color=clrs, alpha=0.6)
                st.pyplot(fig)
                
                # --- 圖表文字敘述說明 ---
                st.markdown("""
                #### 📝 圖表與分析說明
                1. **價格走勢圖 (上圖)**：藍色曲線代表近 40 日收盤價。**紅虛線** 為 AI 預測的隔日壓力位，**綠虛線** 為 AI 預測的隔日支撐位。
                2. **成交量柱狀圖 (下圖)**：紅色柱狀代表當日收紅 K（買氣強），綠色柱狀代表當日收黑 K（拋售強）。
                3. **點位準確性**：Box 內的「達成率」是根據該股過去 60 天符合此 AI 波動特徵的次數計算。
                4. **籌碼校正**：若信心指數權重上升，紅綠虛線會同步上移，代表股價有更高機會突破前高。
                """)
