import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 設定頁面
st.set_page_config(page_title="台股秒級決策助手", layout="centered")

st.title("⚡ 盤中即時量價決策 (優化版)")
st.caption("註：yfinance 免費版在盤中仍有約 15 分鐘延遲，實戰時請對照即時看盤軟體。")

stock_id = st.text_input("輸入台股代碼 (例如: 2330, 8088):")

if stock_id:
    with st.spinner('計算量能倍數與趨勢中...'):
        symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
        
        # 抓取 1分鐘 K線 (盤中最強數據) 與 日線 (算波動率)
        ticker = yf.Ticker(symbol)
        df_1m = ticker.history(interval="1m", period="1d") # 今日分鐘線
        df_daily = ticker.history(period="20d") # 近期日線
        
        if not df_1m.empty and len(df_daily) > 1:
            # 1. 基礎數據提取
            curr_p = df_1m['Close'].iloc[-1]
            open_p = df_1m['Open'].iloc[0]
            prev_c = df_daily['Close'].iloc[-2]
            
            # 2. ATR 波動率計算 (使用日線)
            high_low = df_daily['High'] - df_daily['Low']
            high_cp = np.abs(df_daily['High'] - df_daily['Close'].shift())
            low_cp = np.abs(df_daily['Low'] - df_daily['Close'].shift())
            atr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean().iloc[-1]

            # 3. 量能分析
            curr_v = df_1m['Volume'].sum()
            avg_v = df_daily['Volume'].mean()
            vol_ratio = curr_v / avg_v # 當前成交量佔日均量的比例

            # --- 動態策略邏輯 ---
            st.subheader(f"📊 即時監控：{stock_id}")
            
            # 顯示看板
            m1, m2, m3 = st.columns(3)
            m1.metric("當前價", f"{curr_p:.2f}", f"{((curr_p/prev_c)-1)*100:+.2f}%")
            m2.metric("開盤價", f"{open_p:.2f}", f"跳空 {((open_p/prev_c)-1)*100:+.2f}%")
            m3.metric("量能倍數", f"{vol_ratio:.2f}x", "對比均量")

            st.divider()

            # --- 核心操作建議 (解決你買不到的問題) ---
            if curr_p > open_p and curr_p > prev_c:
                # 情況 A：強勢股 (開高走高或量大)
                st.success("🔥 **多頭攻擊：趨勢強勁**")
                # 強勢時，買點不能設太低，改設在開盤價上方一點點
                st_buy = open_p + (atr * 0.1)
                st_sell = curr_p + (atr * 0.8)
                st.write(f"💡 **買進建議**：觀察 **{st_buy:.2f}** 是否守穩 (開盤價防線)")
                st.write(f"💡 **停利目標**：預估壓力位 **{st_sell:.2f}**")
                
            elif curr_p < prev_c:
                # 情況 B：弱勢股 (破平盤)
                st.error("❄️ **空頭轉弱：不宜進場**")
                st_low_buy = curr_p - (atr * 0.5)
                st.write(f"⚠️ **操作警告**：目前股價在平盤以下，翻紅機率低。")
                st.write(f"💡 **若要低接**：至少等回測至 **{st_low_buy:.2f}** 且出現長下影線。")
            
            else:
                # 情況 C：盤整
                st.info("⚖️ **區間震盪：盤整待變**")
                st.write(f"💡 **操作建議**：在 **{prev_c:.2f}** (平盤) 附近小量試單。")

            # 圖表：今日分鐘線走勢
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_1m.index, df_1m['Close'], color='blue', label="1-min Trend")
            ax.axhline(y=open_p, color='orange', linestyle='--', label="Open")
            ax.axhline(y=prev_c, color='gray', linestyle='--', label="Prev Close")
            ax.set_title("Intraday 1-min Chart")
            ax.legend()
            st.pyplot(fig)

        else:
            st.warning("目前非交易時段或無法獲取分鐘數據。")
