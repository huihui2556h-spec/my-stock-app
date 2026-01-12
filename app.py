import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from datetime import datetime, timedelta

# =========================================================
# 1. 初始化系統配置
# =========================================================
st.set_page_config(page_title="台股 AI 多因子動態回測系統", layout="wide")

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    """頁面導航函數"""
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 籌碼模組：串接 FinMind 資料庫
# =========================================================
def get_institutional_chips(stock_id):
    """
    透過法人買賣超數據計算籌碼修正權重
    """
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取近 14 天資料確保有足夠的交易日
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_dt)
        
        chip_weight = 1.0 
        analysis_note = "籌碼動向：中性震盪"
        
        if not inst_df.empty:
            recent_inst = inst_df.tail(9) 
            net_buy = recent_inst['buy'].sum() - recent_inst['sell'].sum()
            if net_buy > 0:
                chip_weight += 0.018 # 買超權重上調
                analysis_note = "✅ 籌碼動向：法人近期持續加碼"
            elif net_buy < 0:
                chip_weight -= 0.018 # 賣超權重下調
                analysis_note = "⚠️ 籌碼動向：法人近期調節賣出"
        return round(chip_weight, 4), analysis_note
    except:
        return 1.0, "⚠️ 籌碼資料：API 連線中，暫以 1.0 計算"

# =========================================================
# 3. AI 核心引擎：自適應波動預測 (核心添加 FinMind)
# =========================================================
def ai_forecast_engine(df, chip_f=1.0):
    """
    根據波動慣性 (Volatility) 與 籌碼 (Chips) 動態調整預測百分比
    """
    # 計算近 20 日價格標準差 (波動率)
    vol = df['Close'].pct_change().tail(20).std()
    
    # 動態調整分位數：當波動變大，預測區間自動拉寬
    h1_q, l1_q = (0.85, 0.15) if vol > 0.02 else (0.75, 0.25)
    h5_q, l5_q = (0.95, 0.05) if vol > 0.02 else (0.92, 0.08)
    
    # 計算歷史變動百分比
    df_clean = df.tail(80).copy()
    df_clean['h_pct'] = (df_clean['High'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
    df_clean['l_pct'] = (df_clean['Low'] - df_clean['Close'].shift(1)) / df_clean['Close'].shift(1)
    
    # 產出預估百分比 (結合籌碼因子 chip_f)
    h1 = df_clean['h_pct'].quantile(h1_q) * chip_f
    l1 = df_clean['l_pct'].quantile(l1_q) / chip_f
    h5 = df_clean['h_pct'].quantile(h5_q) * chip_f
    l5 = df_clean['l_pct'].quantile(l5_q) / chip_f
    
    return h1, l1, h5, l5

# =========================================================
# 4. 雙向獨立回測引擎：計算每個預估值的準確率
# =========================================================
def multi_period_backtest(df, chip_f):
    """
    模擬過去 20 天，分別計算四個目標的「真實觸及機率」
    """
    test_days = 20
    # 確保資料長度足夠看 T+5 的結果
    hist_data = df.tail(test_days + 60 + 5)
    
    hits = {"h1": 0, "l1": 0, "h5": 0, "l5": 0}
    
    for i in range(test_days):
        # 訓練窗口 (過去 60 天)
        train_window = hist_data.iloc[i : i+60]
        prev_close = hist_data.iloc[i+60-1]['Close']
        
        # 獲取該交易日的預測值
        h1_t, l1_t, h5_t, l5_t = ai_forecast_engine(train_window, chip_f)
        
        # 檢查隔日結果 (T+1)
        day_plus_1 = hist_data.iloc[i+60]
        if day_plus_1['High'] >= prev_close * (1 + h1_t): hits["h1"] += 1
        if day_plus_1['Low'] <= prev_close * (1 + l1_t): hits["l1"] += 1
        
        # 檢查五日內結果 (T+1 ~ T+5)
        window_5d = hist_data.iloc[i+60 : i+65]
        if window_5d['High'].max() >= prev_close * (1 + h5_t): hits["h5"] += 1
        if window_5d['Low'].min() <= prev_close * (1 + l5_t): hits["l5"] += 1
            
    # 計算百分比
    return {k: (v / test_days) * 100 for k, v in hits.items()}

# =========================================================
# 5. UI 介面與圖表渲染
# =========================================================
def render_box(label, price, pct, acc, color="red"):
    """美化顯示盒子"""
    c_code = "#FF4B4B" if color == "red" else "#28A745"
    st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-top: 5px solid {c_code}; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <p style="margin:0; font-size:14px; color:#555; font-weight:bold;">{label}</p>
            <h2 style="margin:5px 0; color:#111; font-size:28px;">{price:.2f}</h2>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{c_code}; font-size:13px; font-weight:bold;">預估振幅 {pct:.2f}%</span>
                <span style="background-color:#eee; padding:2px 6px; border-radius:4px; font-size:13px; font-weight:bold;">🎯 準確率: {acc:.1f}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# 頁面主體
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子動態回測系統 Pro")
    st.markdown("---")
    if st.button("🚀 進入深度分析系統", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 AI 預判與四重達成率回測")
    sid = st.text_input("請輸入台股代碼 (例如: 2330):", key="sid_final")
    
    if sid:
        with st.spinner('AI 正在計算波動慣性與雙向回測數據...'):
            # 獲取最新資料，確保包含當日收盤
            df = yf.download(f"{sid}.TW", period="200d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.ffill()
                
                # --- 核心數據點 ---
                curr_c = float(df['Close'].iloc[-1]) # 最新收盤價
                curr_date = df.index[-1].strftime('%Y-%m-%d')
                
                # 執行籌碼分析與 AI 預測
                chip_f, chip_msg = get_institutional_chips(sid)
                h1, l1, h5, l5 = ai_forecast_engine(df, chip_f)
                
                # 執行四重獨立回測
                bt_results = multi_period_backtest(df, chip_f)

                # --- UI 顯示 ---
                st.subheader(f"🏠 分析標的：{sid} (最新數據日: {curr_date})")
                st.metric("當前收盤基準價", f"{curr_c:.2f}")
                st.info(f"💡 {chip_msg}")

                st.divider()
                st.markdown("### 📅 預估目標與各自回測準確率")
                col1, col2 = st.columns(2)
                with col1:
                    render_box("📈 隔日最高壓力 (T+1)", curr_c*(1+h1), h1*100, bt_results["h1"], "red")
                    render_box("🚩 五日波段高點 (T+5)", curr_c*(1+h5), h5*100, bt_results["h5"], "red")
                with col2:
                    render_box("📉 隔日最低支撐 (T+1)", curr_c*(1+l1), l1*100, bt_results["l1"], "green")
                    render_box("⚓ 五日波段低點 (T+5)", curr_c*(1+l5), l5*100, bt_results["l5"], "green")

                # --- 📉 視覺化圖表 (防亂碼設計) ---
                st.divider()
                st.write("### 📈 波動預測帶視覺化 (Volatility Band)")
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_data = df['Close'].tail(40)
                ax.plot(plot_data.index, plot_data, label="Close Price", color="#1f77b4", lw=2)
                
                # 預測線條 (使用英文標註避免亂碼)
                ax.axhline(y=curr_c*(1+h1), color='red', ls='--', alpha=0.6, label="T+1 Pressure")
                ax.axhline(y=curr_c*(1+h5), color='red', ls='-', alpha=0.3, label="T+5 High")
                ax.axhline(y=curr_c*(1+l1), color='green', ls='--', alpha=0.6, label="T+1 Support")
                
                # 區間填充
                ax.fill_between(plot_data.index, curr_c*(1+l1), curr_c*(1+h1), color='gray', alpha=0.1)
                
                ax.legend(loc='upper left')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                
                st.markdown(f"""
                ---
                #### 📝 中文註解說明：
                1. **最新收盤價基準**：所有預測點位皆以最新的 `{curr_c:.2f}` 為計算起點。
                2. **獨立準確率**：每個盒子右下角的百分比是根據過去 20 天的「實戰命中」情況算出來的。
                   * 如果**上漲準確率高**：代表這支股票近期動能強，高點容易被觸及。
                   * 如果**下跌準確率低**：代表股價近期相對抗跌，不容易回測到支撐位。
                3. **五日波段 (T+5)**：回測邏輯是檢查「預測後的五天內」是否有觸及過目標，適合週轉期較長的交易者。
                """)
