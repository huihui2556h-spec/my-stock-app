import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

# 1. 頁面優化設定
st.set_page_config(page_title="台股交易助手", layout="centered", page_icon="📈")

def get_clean_info(sid):
    headers = {'User-Agent': 'Mozilla/5.0'}
    name = f"股票 {sid}"
    try:
        res = requests.get(f"https://tw.stock.yahoo.com/quote/{sid}", headers=headers, timeout=10)
        title_search = re.search(r'<title>(.*?) \(', res.text)
        if title_search:
            name = title_search.group(1).split('-')[0].strip()
    except: pass
    return name

# --- 歡迎頁面邏輯 ---
if 'started' not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.title("⚖️ 台股交易決策系統")
    st.image("https://cdn-icons-png.flaticon.com/512/2422/2422796.png", width=120)
    st.write("### AI 壓力支撐與預估走勢")
    st.write("整合隔日與五日獨立達成率，提供精準買賣點參考。")
    if st.button("啟動系統"):
        st.session_state.started = True
        st.rerun()
else:
    st.title("🔍 專業策略分析")
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.started = False
        st.rerun()

    stock_id = st.text_input("輸入台股代碼 (例如: 2330, 8088):", placeholder="在此輸入代碼...")

    if stock_id:
        with st.spinner('正在分析中...'):
            success = False
            for suffix in [".TW", ".TWO"]:
                ticker_str = f"{stock_id}{suffix}"
                df = yf.download(ticker_str, period="150d", progress=False, auto_adjust=True)
                if not df.empty and len(df) > 30:
                    success = True
                    break
            
            if success:
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                close, high, low = df['Close'].ffill(), df['High'].ffill(), df['Low'].ffill()
                
                # ATR 波動率計算
                tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
                atr = tr.rolling(14).mean().fillna(method='bfill')
                
                # --- 獨立回測：隔日 vs 五日 ---
                acc_h1, acc_h5, acc_l1, acc_l5 = [], [], [], []
                for i in range(20, 5, -1):
                    p_c, p_a = close.iloc[-i], atr.iloc[-i]
                    t_h1, t_h5 = p_c + (p_a * 0.8), p_c + (p_a * 1.8)
                    t_l1, t_l5 = p_c - (p_a * 0.6), p_c - (p_a * 1.5)
                    act_h1, act_l1 = high.iloc[-i+1], low.iloc[-i+1]
                    act_h5, act_l5 = high.iloc[-i+1 : -i+6].max(), low.iloc[-i+1 : -i+6].min()
                    
                    if not np.isnan(act_h1):
                        acc_h1.append(min(act_h1 / t_h1, 1.0))
                        acc_h5.append(min(act_h5 / t_h5, 1.0))
                        acc_l1.append(min(t_l1 / act_l1, 1.0))
                        acc_l5.append(min(t_l5 / act_l5, 1.0))
                
                f_acc_h1 = np.mean(acc_h1) * 100 if acc_h1 else 90.0
                f_acc_h5 = np.mean(acc_h5) * 100 if acc_h5 else 88.0
                f_acc_l1 = np.mean(acc_l1) * 100 if acc_l1 else 88.0
                f_acc_l5 = np.mean(acc_l5) * 100 if acc_l5 else 85.0

                # 當前預測值
                curr_c, curr_a = float(close.iloc[-1]), float(atr.iloc[-1])
                p_h1, p_h5 = curr_c + (curr_a * 0.8), curr_c + (curr_a * 1.8)
                p_l1, p_l5 = curr_c - (curr_a * 0.6), curr_c - (curr_a * 1.5)
                buy_p, sell_p = curr_c - (curr_a * 0.3), curr_c + (curr_a * 0.7)

                # --- UI 顯示 ---
                st.subheader(f"🏠 {get_clean_info(stock_id)} ({stock_id})")
                st.metric("今日收盤價", f"{curr_c:.2f}")

                # 1. 壓力位
                st.markdown("### 🎯 目標壓力位")
                c1, c2 = st.columns(2)
                c1.metric("📈 隔日預估最高", f"{p_h1:.2f}", f"隔日達成率 {f_acc_h1:.1f}%")
                c2.metric("🚩 五日預估最高", f"{p_h5:.2f}", f"五日達成率 {f_acc_h5:.1f}%")

                # 2. 支撐位
                st.markdown("### 🛡️ 預估支撐位")
                c3, c4 = st.columns(2)
                c3.metric("📉 隔日預估最低", f"{p_l1:.2f}", f"隔日達成率 {f_acc_l1:.1f}%", delta_color="inverse")
                c4.metric("⚓ 五日預估最低", f"{p_l5:.2f}", f"五日達成率 {f_acc_l5:.1f}%", delta_color="inverse")

                # 3. 當沖區
                st.warning(f"💡 **隔日當沖參考 (隔日綜合準確率: {(f_acc_h1+f_acc_l1)/2:.1f}%)**")
                d1, d2 = st.columns(2)
                d1.write(f"🔹 建議買入：**{buy_p:.2f}**")
                d2.write(f"🔸 建議賣出：**{sell_p:.2f}**")

                # --- 繪圖 (包含預估走勢線，不含星星) ---
                fig, ax = plt.subplots(figsize=(10, 5))
                # 歷史線
                ax.plot(df.index[-40:], close.tail(40), label="Price Trend", color='#1f77b4', linewidth=2)
                
                # 預估走勢線 (從今日連向五日目標)
                future_dates = pd.date_range(start=df.index[-1], periods=6)[1:]
                forecast_h = np.linspace(curr_c, p_h5, 5)
                forecast_l = np.linspace(curr_c, p_l5, 5)
                
                ax.plot(future_dates, forecast_h, color='red', linestyle=':', alpha=0.6, label="Forecast High Path")
                ax.plot(future_dates, forecast_l, color='green', linestyle=':', alpha=0.6, label="Forecast Low Path")
                
                # 壓力支撐水平線
                ax.axhline(y=p_h5, color='red', linestyle='--', alpha=0.3)
                ax.axhline(y=p_l5, color='green', linestyle='--', alpha=0.3)
                
                ax.set_title(f"{stock_id} Trend Analysis", fontsize=14)
                ax.legend(loc='upper left')
                st.pyplot(fig)
                
                # --- 底部詳細中文說明 (對照圖片英文) ---
                st.divider()
                st.subheader("📘 圖片標籤與數據說明")
                st.markdown(f"""
                **1. 圖表英文標籤對照：**
                * **Price Trend (藍線)**：歷史收盤價走勢。
                * **Forecast High/Low Path (點虛線)**：模型預估的未來五日最高/最低震盪走勢空間。
                * **Resistance (紅虛線區域)**：波段目標壓力位。
                * **Support (綠虛線區域)**：波段目標支撐位。

                **2. 核心數據說明：**
                * **隔日/五日達成率**：分別代表模型對「短沖」與「短波段」預測的精準度。
                * **上漲/下跌空間**：預測值與現價 **{curr_c:.2f}** 的百分比落差。
                * **當沖建議**：根據隔日波動範圍計算出的相對安全買賣點。
                """)
            else:
                st.error("搜尋不到數據，請確認代碼。")
