import datetime
import pytz

# --- 模式 B: 盤中即時決策修復版 ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
    st.title("⚡ 盤中即時量價建議")
    
    # 判斷是否為交易時段 (台灣時間)
    tz = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tz)
    is_open = now.weekday() < 5 and 9 <= now.hour < 14 # 簡化判斷 9:00-14:00

    stock_id = st.text_input("輸入代碼 (如: 4979):", key="rt_id")
    if stock_id:
        with st.spinner('正在計算即時買賣建議...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            # 抓取 1分鐘線 (即時) 與 日線 (算波動基準)
            ticker = yf.Ticker(symbol)
            df_rt = ticker.history(period="1d", interval="1m")
            df_hist = ticker.history(period="5d")
            
            if not df_rt.empty and not df_hist.empty:
                curr_p = float(df_rt['Close'].iloc[-1])
                open_p = float(df_rt['Open'].iloc[0])
                prev_c = float(df_hist['Close'].iloc[-2])
                
                # 計算即時波動基準 (ATR 估計)
                atr_est = (df_hist['High'] - df_hist['Low']).mean()

                st.subheader(f"📊 {get_clean_info(stock_id)}")
                if not is_open:
                    st.info(f"📅 目前非交易時段。顯示數據為最後交易日收盤資訊。")
                
                c1, c2 = st.columns(2)
                c1.metric("當前成交價", f"{curr_p:.2f}", f"{((curr_p/prev_c)-1)*100:+.2f}%")
                c2.metric("今日開盤價", f"{open_p:.2f}", f"跳空 {((open_p/prev_c)-1)*100:+.2f}%")

                st.divider()
                st.markdown("### 🏹 盤中即時操盤建議")
                
                # 動態計算買賣建議價
                buy_strong = open_p - (atr_est * 0.1) # 強勢買點(守開盤)
                buy_low = curr_p - (atr_est * 0.4)    # 低接買點(超跌)
                sell_target = curr_p + (atr_est * 0.6) # 盤中壓力賣點

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    if curr_p >= open_p:
                        st.success(f"🔥 **強勢建議買入**：{buy_strong:.2f}")
                    else:
                        st.warning(f"❄️ **弱勢低接買入**：{buy_low:.2f}")
                
                with col_s2:
                    st.info(f"🔸 **建議賣出點**：{sell_target:.2f}")

                st.caption(f"註：買賣建議根據盤中波動率自動調整，建議配合量能觀察。")
            else:
                st.error("找不到該股票數據，可能代碼錯誤或該股今日無交易。")
