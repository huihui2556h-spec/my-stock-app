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
# 1. 系統環境設定 (設定網頁標題與顯示模式)
# =========================================================
st.set_page_config(page_title="台股 AI 高精度預估系統", layout="centered")

# 初始化頁面導航狀態，預設顯示為 'home' (首頁)
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    """【導航函數】處理頁面切換並重新渲染頁面"""
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 核心運算：多因子權重與誤差補償邏輯
# =========================================================

def get_error_bias(df, days=10):
    """
    【動態誤差修正】提升準確度的關鍵！
    目的：計算過去 10 天 AI 預估點位與實際高低價的偏差。
    邏輯：如果最近股價波動強於預期，系統會自動回傳一個加乘權重(Bias)，補強後續預估。
    """
    try:
        temp = df.copy().tail(days + 15)
        # 計算 14 日平均真實波幅 (ATR)
        temp['ATR'] = (temp['High'] - temp['Low']).rolling(14).mean()
        biases = []
        for i in range(1, days + 1):
            prev_c = temp['Close'].iloc[-i-1] # 前一日收盤
            prev_atr = temp['ATR'].iloc[-i-1] # 前一日 ATR
            actual_h = temp['High'].iloc[-i] # 今日實際最高價
            # 如果 ATR 存在，則計算 (實際最高價 / AI理論最高點) 的比率
            if prev_atr > 0:
                biases.append(actual_h / (prev_c + prev_atr * 0.85))
        # 取平均偏差值，若無數據則維持 1.0 (不修正)
        return np.mean(biases) if biases else 1.0
    except:
        return 1.0

def get_chip_factor(stock_id):
    """
    【FinMind 籌碼因子】法人慣性修正 (2026-01-12 指令)
    目的：獲取三大法人近 5 日買賣超，決定價格是向上還是向下修正。
    """
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取最近 15 天數據，確保扣除例假日後有足夠 5 天交易日
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            # 計算五日合計買賣淨額
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            # 若合計為買超，給予 1.025 的向上慣性加成；反之給予 0.975
            return (1.025, "✅ 籌碼面：法人偏多 (近五日合計買超)") if net_buy > 0 else (0.975, "⚠️ 籌碼面：法人偏空 (近五日合計賣超)")
    except:
        pass
    return 1.0, "ℹ️ 籌碼面：中性 (數據連線中)"

def calculate_real_accuracy(df, atr_factor, side='high'):
    """
    【高精度回測】計算過去 60 天 AI 點位的命中機率
    目的：提供一個「信任指標」，讓使用者知道該點位在歷史上被觸及的頻率。
    """
    try:
        temp = df.copy().ffill()
        # 處理 Yahoo Finance 的多重索引格式 (MultiIndex)
        if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.get_level_values(0)
        backtest_days = min(len(temp) - 15, 60) # 扣除 ATR 預熱期後回測 60 天
        hits = 0
        temp['ATR_CALC'] = (temp['High'] - temp['Low']).rolling(14).mean()
        
        for i in range(1, backtest_days + 1):
            idx = -i
            p_close, p_atr = temp['Close'].iloc[idx-1], temp['ATR_CALC'].iloc[idx-1]
            actual = temp['High'].iloc[idx] if side == 'high' else temp['Low'].iloc[idx]
            # 模擬歷史預測公式
            pred = p_close + (p_atr * atr_factor) if side == 'high' else p_close - (p_atr * atr_factor)
            # 判斷當天行情是否達標
            if (side == 'high' and actual >= pred) or (side == 'low' and actual <= pred): 
                hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

# =========================================================
# 3. 網頁呈現邏輯
# =========================================================

if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("已整合：FinMind 籌碼、1/5/10日多維度預估、誤差修正模型")
    if st.button("🚀 啟動 AI 分析儀", use_container_width=True): navigate_to("forecast")

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 深度預估分析報告")
    sid = st.text_input("請輸入股票代碼 (例: 2330):", key="fc_id")

    if sid:
        with st.spinner('AI 正在交叉分析因子並執行誤差補償回測...'):
            # 1. 下載數據 (自動嘗試上市櫃後綴)
            df = None
            for suf in [".TW", ".TWO"]:
                tmp = yf.download(f"{sid}{suf}", period="200d", progress=False)
                if not tmp.empty: df = tmp; break
            
            if df is not None:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # 2. 獲取計算因子
                chip_f, chip_msg = get_chip_factor(sid) # 籌碼因子
                err_bias = get_error_bias(df)           # 誤差修正因子 (提升準確度核心)
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1] # 最新 ATR
                curr_c = float(df['Close'].iloc[-1])    # 最新收盤價
                
                # 綜合修正權重
                final_bias = chip_f * err_bias
                
                # 3. 計算 1日、5日、10日 預估壓力與支撐點位
                ph1, pl1 = curr_c + (atr * 0.85 * final_bias), curr_c - (atr * 0.65 / final_bias)
                ph5, pl5 = curr_c + (atr * 1.90 * final_bias), curr_c - (atr * 1.60 / final_bias)
                ph10, pl10 = curr_c + (atr * 2.80 * final_bias), curr_c - (atr * 2.30 / final_bias)

                # 4. 介面呈現
                st.subheader(f"🏠 分析標的: {sid}")
                st.info(f"{chip_msg} | 歷史偏誤補償: {err_bias:.3f}")
                
                # 分頁切換顯示不同時間維度
                t1, t5, t10 = st.tabs(["🎯 隔日預估", "🚩 五日波段", "⚓ 十日長波段"])
                
                def show_box(price, side, factor, label):
                    """內部美化顯示函數"""
                    acc = calculate_real_accuracy(df, factor, side)
                    color = "#FF4B4B" if side == "high" else "#28A745"
                    st.markdown(f"""
                        <div style='border-left:5px solid {color}; padding:15px; background:#f0f2f6; margin-bottom:10px; border-radius:5px;'>
                            <p style='margin:0; font-size:14px; color:#666;'>{label}</p>
                            <h2 style='margin:0; color:#333;'>{price:.2f}</h2>
                            <p style='margin:0; font-size:12px; color:#888;'>↳ 歷史達成率: <b>{acc:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                with t1:
                    c1, c2 = st.columns(2)
                    with c1: show_box(ph1, "high", 0.85, "📈 隔日最高預估")
                    with c2: show_box(pl1, "low", 0.65, "📉 隔日最低預估")
                with t5:
                    c1, c2 = st.columns(2)
                    with c1: show_box(ph5, "high", 1.90, "📈 五日最高壓力")
                    with c2: show_box(pl5, "low", 1.60, "📉 五日最低支撐")
                with t10:
                    c1, c2 = st.columns(2)
                    with c1: show_box(ph10, "high", 2.80, "📈 十日波段頂部")
                    with c2: show_box(pl10, "low", 2.30, "📉 十日波段底部")

                # 5. 明日當沖建議點位 (結合 ATR 與因子修正)
                st.divider()
                st.markdown("### 🏹 明日當沖/短線策略指引")
                d1, d2, d3 = st.columns(3)
                d1.info(f"🔹 **追多買點**\n\n**{curr_c + (atr * 0.15):.2f}**")
                d2.error(f"🔹 **支撐低接**\n\n**{curr_c - (atr * 0.45):.2f}**")
                d3.success(f"🔸 **短線目標**\n\n**{curr_c + (atr * 0.75):.2f}**")

                # 6. 價量趨勢圖 (英文標籤避免亂碼)
                st.divider()
                st.write("📊 價量趨勢與 AI 波段參考圖 (Price & Volume)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.tail(40).index, df.tail(40)['Close'], label="Close Price", lw=2)
                # 畫出五日壓力與支撐虛線
                ax.axhline(y=ph5, color='red', ls='--', alpha=0.5, label="5D Resistance")
                ax.axhline(y=pl5, color='green', ls='--', alpha=0.5, label="5D Support")
                ax.legend(loc='upper left')
                st.pyplot(fig)
                st.caption("註：紅虛線為預估五日壓力位，綠虛線為預估五日支撐位。")
            else:
                st.error("❌ 無法抓取數據，請檢查代碼是否正確。")
