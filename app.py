# --- 模式 B: 盤中即時決策 (含未開盤通知與建議價) ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
    
    st.title("⚡ 盤中即時量價建議")
    
    # 1. 取得台灣時間與判斷開盤狀態
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tw_tz)
    
    # 判斷邏輯
    is_weekday = now.weekday() < 5
    is_market_time = 9 <= now.hour < 14  # 簡單判斷 09:00 - 13:59
    
    stock_id = st.text_input("請輸入台股代碼 (如: 2330, 4979):", key="rt_id")
    
    if stock_id:
        # 2. 顯示開盤狀態通知
        if not is_weekday:
            st.warning(f"🔔 【目前未開盤】今天為週末，以下顯示數據為前一交易日資訊。")
        elif now.hour < 9:
            st.info(f"🔔 【目前未開盤】今日台股尚未開盤（09:00 開盤），以下為預估建議價。")
        elif now.hour >= 14:
            st.info(f"🔔 【今日已收盤】目前顯示今日結算數據與隔日建議。")

        with st.spinner('正在計算買賣建議價...'):
            symbol = f"{stock_id}.TW" if int(stock_id) < 10000 else f"{stock_id}.TWO"
            # 抓取數據 (1d 用於即時, 5d 用於計算 ATR 波動率)
            df_rt = yf.download(symbol, period="1d", interval="1m", progress=False)
            df_hist = yf.download(symbol, period="5d", progress=False)
            
            if not df_rt.empty and not df_hist.empty:
                # 處理資料格式
                if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
                if isinstance(df_hist.columns, pd.MultiIndex): df_hist.columns = df_hist.columns.get_level_values(0)

                curr_p = float(df_rt['Close'].iloc[-1])
                open_p = float(df_rt['Open'].iloc[0])
                prev_c = float(df_hist['Close'].iloc[-2])
                
                # 計算波動基準 (ATR 簡化版)
                atr_est = (df_hist['High'] - df_hist['Low']).mean()

                st.subheader(f"📊 {get_clean_info(stock_id)} ({symbol})")
                
                # 3. 顯示當前價與開盤價
                c1, c2 = st.columns(2)
                c1.metric("當前/最後成交價", f"{curr_p:.2f}", f"{((curr_p/prev_c)-1)*100:+.2f}%")
                c2.metric("今日開盤價", f"{open_p:.2f}", f"跳空 {((open_p/prev_c)-1)*100:+.2f}%")

                # 4. 當沖建議價格區塊 (核心需求)
                st.divider()
                st.markdown("### 🏹 當沖建議買賣價格")
                
                # 根據波動率計算數值
                buy_strong = open_p - (atr_est * 0.1)  # 強勢買點 (回踩開盤)
                buy_low = curr_p - (atr_est * 0.45)     # 低接買點 (超跌)
                sell_target = curr_p + (atr_est * 0.75) # 盤中壓力賣點

                d1, d2, d3 = st.columns(3)
                d1.write("**🔹 強勢買入價**")
                d1.info(f"{buy_strong:.2f}")
                
                d2.write("**🔹 低接買入價**")
                d2.error(f"{buy_low:.2f}")
                
                d3.write("**🔸 建議賣出價**")
                d3.success(f"{sell_target:.2f}")

                st.caption(f"💡 提醒：若目前為「未開盤」狀態，開盤價將以最後交易日資訊計算。實際操作請對照 09:00 後的真實開盤價。")
            else:
                st.error("無法取得該代碼之數據，請檢查輸入是否正確。")
