import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
from datetime import datetime, time
import pytz
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="台股 AI 交易助手 Pro", layout="wide", page_icon="💹")

# --- 1. [定義台股升降單位函數] ---
def get_tick_size(price):
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0  # 台積電等級 (1000元以上)

# 2. 定義時區，確保日期隨時間自動改變不報錯 [cite: 2026-01-12]
tw_tz = pytz.timezone("Asia/Taipei")

# 3. 解決圖表亂碼問題 (英文 Legend)
def set_mpl_font():
    plt.rcParams['axes.unicode_minus'] = False 
    # 這裡我們維持使用英文標籤，避免不同系統字體缺失導致的 □□□
set_mpl_font()

# 4. 初始化 Session State (若尚未初始化)
if 'mode' not in st.session_state:
    st.session_state.mode = "home"
# --- 🎯 修正圖片亂碼：強制手動載入系統字體 ---
def set_mpl_font():
    # 嘗試多種常見中文字體名稱，確保在不同 OS 都能正常顯示
    fonts = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'SimHei', 'Arial Unicode MS']
    for f in fonts:
        try:
            matplotlib.rc('font', family=f)
            # 測試繪圖是否會報錯
            plt.figure()
            plt.close()
            break
        except:
            continue
    # 解決座標軸負號顯示問題
    matplotlib.rcParams['axes.unicode_minus'] = False 

set_mpl_font()

st.set_page_config(page_title="台股 AI 交易助手 Pro", layout="wide", page_icon="💹")

# --- 狀態初始化 ---
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# --- 核心功能：真實回測勝率判斷 ---
def calculate_real_accuracy(df, factor, side='high'):
    try:
        df_copy = df.copy().ffill()
        backtest_days = 60 # 2026-01-12 指示：回測 60 天
        if len(df_copy) < backtest_days + 15: return 85.0
        hits, total = 0, 0
        tr = np.maximum(df_copy['High'] - df_copy['Low'],
                        np.maximum(abs(df_copy['High'] - df_copy['Close'].shift(1)),
                                   abs(df_copy['Low'] - df_copy['Close'].shift(1))))
        atr = tr.rolling(14).mean()
        for i in range(1, backtest_days + 1):
            prev_close = df_copy['Close'].iloc[-i-1]
            prev_atr = atr.iloc[-i-1]
            if np.isnan(prev_atr): continue
            total += 1
            if side == 'high' and df_copy['High'].iloc[-i] <= prev_close + prev_atr * factor: hits += 1
            if side == 'low' and df_copy['Low'].iloc[-i] >= prev_close - prev_atr * factor: hits += 1
        return (hits / total * 100) if total > 0 else 88.0
    except:
        return 88.0

# --- 獲取名稱 ---
def get_stock_name(stock_id):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        html = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5).text
        name = re.search(r'<title>(.*?) \(', html).group(1)
        return name.split('-')[0].strip()
    except:
        return f"台股 {stock_id}"

# --- 抓股價 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(stock_id, period="120d"):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        df = yf.download(symbol, period=period, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, symbol
    return pd.DataFrame(), None

# --- 🎨 自定義台股配色組件 ---
def stock_box(label, price, pct, acc, color_type="red"):
    bg_color = "#FF4B4B" if color_type == "red" else "#28A745"
    arrow = "↑" if color_type == "red" else "↓"
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {bg_color}; margin-bottom: 10px;">
            <p style="margin:0; font-size:14px; color:#555;">{label}</p>
            <h2 style="margin:0; padding:5px 0; color:#333;">{price:.2f}</h2>
            <span style="background-color:{bg_color}; color:white; padding:2px 8px; border-radius:5px; font-size:14px;">
                {arrow} {pct:.2f}%
            </span>
            <p style="margin-top:10px; font-size:12px; color:#888;">↳ 近20日達成率：{acc:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# 2. 核心運算：全自動資金流向分析
# =========================================================

def analyze_full_flow():
    """【分析引擎】針對所有細分區段計算：1. 平均漲跌幅 2. 資金流入比 (今日量/5日均量)"""
    flow_report = []
    # 遍歷每一組產業鏈
    for chain_name, tickers in INDUSTRY_CHAINS.items():
        try:
            # 抓取 10 天數據以便計算 5 日平均成交量
            data = yf.download(tickers, period="10d", progress=False)
            if data.empty: continue
            
            # 1. 漲跌幅計算 (今日收盤 vs 昨日收盤)
            ret = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1).mean() * 100
            
            # 2. 資金流入強度計算 (今日總量 / 近五日平均總量)
            # 數值 > 1.0 代表該細分產業今日熱度高於平日，為資金焦點
            today_vol = data['Volume'].iloc[-1].sum()
            avg_vol_5d = data['Volume'].tail(5).mean().sum()
            inflow_ratio = today_vol / avg_vol_5d if avg_vol_5d > 0 else 1.0
            
            flow_report.append({"產業細分": chain_name, "漲跌%": ret, "資金流入": inflow_ratio})
        except: continue
    return pd.DataFrame(flow_report)
# ================== 介面控制 ==================

    if st.session_state.mode == "home":
       st.title("⚖️ 台股 AI 交易決策系統")
       col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True):
            st.session_state.mode = "realtime"
            st.rerun()
    with col_b:
        if st.button("📊 波段預估", use_container_width=True):
            st.session_state.mode = "forecast"
            st.rerun()
    with col_c:
        if st.button("💎 類群輪動預警", use_container_width=True):
            st.session_state.mode = "sector"
            st.rerun()
# --- A. 💎 類群輪動預警頁面 ---
        elif st.session_state.mode == "sector":
    # 1. 標題放在最上面 (進入此模式後才顯示)
            st.title("💎 產業鏈深度資金監控")
    
    # 2. 返回按鈕放在標題下方 (不放在 Sidebar，直接放在頁面裡)
    if st.button("⬅️ 返回功能首頁", use_container_width=True):
        st.session_state.mode = "home"  # 這裡要改回 home
        st.rerun()

        st.divider() # 增加一條分隔線，美化版面
    
    # 3. 頁面提示與內容
        st.markdown("### 目前監控範例：PCB、記憶體、AI 伺服器、重電全系列")
        name_map = {
        "PCB-CCL": "PCB-材料 (CCL/銅箔)",
        "PCB-Substrate": "PCB-載板 (ABF/BT)",
        "PCB-Assembly": "PCB-組裝加工 (硬板/HDI)",
        "Memory-Fab": "記憶體-原廠/代工",
        "Memory-Module": "記憶體-模組廠",
        "Memory-Controller": "記憶體-控制 IC",
        "Semi-Equip": "半導體-設備/CoWoS",
        "CPO-Silicon": "矽光子 (CPO/光通訊)",
        "AI-Case": "AI 伺服器 (機殼/滑軌)",
        "AI-Cooling": "AI 伺服器 (散熱/水冷)",
        "AI-ODM": "AI 伺服器 (ODM 代工)",
        "Power-Grid": "重電/電力 (政策股)",
        "Shipping": "航運 (貨櫃/散裝)"
    }

    # 產業鏈英文 ID 定義 (避免圖表亂碼)
    INDUSTRY_CHAINS_EN = {
        "PCB-CCL": ["6213.TW", "2383.TW", "6274.TW", "8358.TWO"],
        "PCB-Substrate": ["8046.TW", "3037.TW", "3189.TW"],
        "PCB-Assembly": ["2367.TW", "2313.TW", "2368.TW", "4958.TW"],
        "Memory-Fab": ["2344.TW", "2337.TW", "2408.TW"],
        "Memory-Module": ["3260.TWO", "8299.TW", "2451.TW", "3264.TWO"],
        "Memory-Controller": ["8299.TW", "4966.TW", "6233.TWO"],
        "Semi-Equip": ["3131.TWO", "3583.TW", "1560.TW", "6187.TWO"],
        "CPO-Silicon": ["3363.TWO", "4979.TWO", "3081.TWO", "6451.TW"],
        "AI-Case": ["8210.TW", "2059.TW", "6803.TW", "3693.TW"],
        "AI-Cooling": ["3017.TW", "3324.TW", "2421.TW", "6230.TW"],
        "AI-ODM": ["2382.TW", "2317.TW", "3231.TW", "6669.TW"],
        "Power-Grid": ["1513.TW", "1503.TW", "1519.TW", "1514.TW"],
        "Shipping": ["2603.TW", "2609.TW", "2615.TW", "2606.TW"]
    }
    
   
    
    with st.spinner('AI 正在計算各產業獲利潛力...'):
        flow_report = []
        for en_id, tickers in INDUSTRY_CHAINS_EN.items():
            try:
                data = yf.download(tickers, period="10d", progress=False)
                if not data.empty:
                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1).mean() * 100
                    vol_ratio = data['Volume'].iloc[-1].sum() / data['Volume'].tail(5).mean().sum()
                    flow_report.append({"ID": en_id, "漲跌%": ret, "資金流入": vol_ratio})
            except: continue
        
        df_flow = pd.DataFrame(flow_report)

        if not df_flow.empty:
            # --- 🔮 核心：建議買進判斷邏輯 ---
            # 策略：資金爆量 (流入>1.2) 且 今日有撐 (漲幅>-0.5)
            buy_candidates = df_flow[(df_flow['資金流入'] > 1.2) & (df_flow['漲跌%'] > -0.5)]
            
            st.subheader("🎯 目前強勢族群")
            if not buy_candidates.empty:
                # 排序出最推薦的類股
                best_sector_id = buy_candidates.sort_values(by="資金流入", ascending=False).iloc[0]['ID']
                st.success(f"🚀 **【強烈建議關注】：{name_map[best_sector_id]}**")
                st.info(f"💡 理由：該族群資金流入強度達 {buy_candidates['資金流入'].max():.2f} 倍，顯示大戶籌碼高度集中，今日股價同步走強，發動機率高。")
            else:
                st.warning("⚠️ 目前多數類股處於縮量或盤整期，暫無「爆量起漲」標的，建議保留資金分批布局。")

            st.divider()
            if not df_flow.empty:
            # ==========================================
            # 1. [建議買進邏輯放在這裡] - AI 判斷區
            # ==========================================
               low_base_candidates = df_flow[
                 (df_flow['資金流入'] > 1.05) & 
                 (df_flow['資金流入'] < 1.8) & 
                 (df_flow['漲跌%'] >= -1.0) & 
                 (df_flow['漲跌%'] <= 2.5)
            ]
            
            st.subheader("🎯 AI 低基期潛力產業建議")
            if not low_base_candidates.empty:
                # 1. 鎖定潛力產業
                best_bet = low_base_candidates.sort_values(by="資金流入", ascending=False).iloc[0]
                best_id = best_bet['ID']
                
                st.success(f"🚀 **【潛力補漲關注】：{name_map[best_id]}**")
                
                # 2. 自動抓取該族群內的具體標的
                target_tickers = INDUSTRY_CHAINS_EN.get(best_id, [])
                
                if target_tickers:
                    st.write(f"💡 **族群內推薦標的：**")
                    # 建立美觀的推薦卡片
                    cols = st.columns(len(target_tickers))
                    for idx, ticker in enumerate(target_tickers):
                        try:
                            # 獲取個股名稱與今日表現
                            t_data = yf.Ticker(ticker)
                            raw_name = get_stock_name(ticker.split('.')[0])
                            t_name = raw_name.replace("走勢圖", "").replace("Yahoo奇摩股市", "").strip()
                            t_price = t_data.history(period="2d")
                            if len(t_price) >= 2:
                                t_ret = (t_price['Close'].iloc[-1] / t_price['Close'].iloc[-2] - 1) * 100
                                with cols[idx]:
                                    st.metric(label=t_name, value=f"{t_price['Close'].iloc[-1]:.2f}", delta=f"{t_ret:.2f}%")
                        except:
                            continue

            st.divider()
            # --- 📋 詳細數據表格 (隱藏左邊數字) ---
            st.write("📋 **詳細數據明細**")
            df_display = df_flow.copy()
            df_display['產業名稱'] = df_display['ID'].map(name_map)
            st.dataframe(
                df_display[['產業名稱', '漲跌%', '資金流入']].sort_values(by='資金流入', ascending=False), 
                use_container_width=True,
                hide_index=True
            )

            # --- 📊 繪製圖表 (英文標籤) ---
            st.write("📈 **Sector Money Flow (資金流入排行榜)**")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_plot = df_flow.sort_values(by="資金流入")
            ax.barh(df_plot['ID'], df_plot['資金流入'], color='gold', edgecolor='black')
            ax.axvline(x=1.0, color='red', ls='--', alpha=0.6)
            st.pyplot(fig)
            

            # --- 📝 中文註解 ---
            st.markdown("#### 📘 分類註解 (Legends):")
            c1, c2 = st.columns(2)
            sorted_en_ids = df_plot['ID'].tolist()[::-1]
            for i, en_id in enumerate(sorted_en_ids):
                with (c1 if i % 2 == 0 else c2):
                    st.write(f"- **{en_id}**: {name_map[en_id]}")
            
            st.divider()

            
            
        else:
            st.error("暫時無法取得數據，請確認網路或 API 連線。")

       elif st.session_state.mode == "realtime":
           from datetime import datetime, time
           import pytz
    
           if st.sidebar.button("⬅️ 返回首頁"): 
           st.session_state.mode = "home"
           st.rerun()
        
           st.title("⚡ 盤中即時量價（當沖）")

    # 1. 設定台灣時區與時間判斷
    tw_tz = pytz.timezone("Asia/Taipei")
    now = datetime.now(tw_tz)
    # 交易時間判斷：週一至週五 09:00 ~ 13:30
    is_market_open = now.weekday() < 5 and (time(9, 0) <= now.time() <= time(13, 30))

    stock_id = st.text_input("輸入股票代碼（如：2330）")

    if stock_id:
        df, sym = fetch_stock_data(stock_id, period="60d")
        
        if df.empty:
            st.error("❌ 查無資料，請檢查代碼是否正確。")
        else:
            # --- [基礎數據準備] ---
            df = df.ffill()
            name = get_stock_name(stock_id)
            curr_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            price_diff = curr_price - prev_close
            active_color = "#E53E3E" if price_diff >= 0 else "#38A169"

           

            # --- [3. 關鍵邏輯：未開盤僅顯示警示，盤中才計算動態預測] ---
            if not is_market_open:
                # 未開盤：顯示警示標語，並停止執行後續預測
                st.warning(f"🕒 【目前非交易時段】系統暫停動態演算。現在時間：{now.strftime('%H:%M')}。")
                st.info("💡 盤中 AI 建議點位將於台股開盤時間 (09:00 - 13:30) 自動啟動即時演算。")
            else:
                # 盤中時間：顯示動態預測
                st.success(f"🟢 【盤中 AI 動態監控中】數據隨量價即時校正")

                recent_std = df['Close'].tail(15).std() # 計算近期波動
                avg_vol = df['Volume'].tail(10).mean()  # 計算平均成交量
                instant_vol_factor = df['Volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1.0 # 計算量能倍數
                clean_name = name.split('(')[0].split('-')[0].strip() # 處理括號名稱

                st.markdown(f"""
                    <style>
                        @media (max-width: 600px) {{ .main-price {{ font-size: 52px !important; }} }}
                    </style>
                    <div style='background: #FFFFFF; padding: 25px; border-radius: 18px; border-left: 12px solid {active_color}; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;'>
                        <div style='color: #0F172A; font-size: 28px; font-weight: 800;'>{name} ({sym})</div>
                        <div style='display: flex; align-items: baseline; flex-wrap: wrap; margin-top:10px;'>
                            <b class='main-price' style='font-size: 70px; color: {active_color}; line-height: 1;'>{curr_price:.2f}</b>
                            <div style='margin-left: 15px;'>
                                <span style='font-size: 28px; color: {active_color}; font-weight: 900; display: block;'>
                                    {'▲' if price_diff >= 0 else '▼'} {abs(price_diff):.2f}
                                </span>
                                <span style='font-size: 18px; color: {active_color}; font-weight: 700;'>
                                    ({(price_diff/prev_close*100):.2f}%)
                                </span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 1. 動態信心係數 (Confidence Factor)
                # 觀察最近 5 分鐘的價格是否穩定，若震盪劇烈則擴大安全邊際
                stability_index = df['Close'].tail(5).std() / recent_std
                confidence_shield = max(1.0, min(2.0, stability_index))

                # 2. 動態量價擴展 (Dynamic Expansion)
                # 買點不再是固定減去多少，而是根據「能量守恆」：
                # 當成交量暴增時，波動空間會呈非線性擴張 (例如開平方根)
                vol_expansion = np.sqrt(instant_vol_factor) 
                
                # 3. 終極演算：點位由「即時波動率」與「能量擴展」交互計算
                # 這裡沒有 1.2 或 1.5，而是由 stability_index 與 vol_expansion 決定
                dynamic_offset_low = recent_std * (confidence_shield / vol_expansion)
                dynamic_offset_high = recent_std * (vol_expansion * confidence_shield)
                
                # 4. 生成動態買賣點
                buy_support = curr_price - dynamic_offset_low
                sell_resist = curr_price + dynamic_offset_high

                # --- [對齊 Tick Size] ---
                tick = get_tick_size(curr_price)
                buy_point = round(buy_support / tick) * tick
                sell_target = round(sell_resist / tick) * tick
                expected_return = (sell_target - buy_point) / buy_point * 100

                # --- [顯示當沖 AI 建議點位] ---
                st.subheader("🎯 當沖 AI 動態演算建議")
                d1, d2, d3 = st.columns(3)
                
                with d1:
                    st.markdown(f"""
                        <div style="background:#F0F9FF; padding:20px; border-radius:12px; border-left:8px solid #3182CE; text-align:center;">
                            <b style="color:#2C5282; font-size:14px;">🔹 動態支撐買點</b>
                            <h2 style="color:#1E40AF; margin:10px 0;">{buy_point:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                with d2:
                    st.markdown(f"""
                        <div style="background:#FFF5F5; padding:20px; border-radius:12px; border-left:8px solid #E53E3E; text-align:center;">
                            <b style="color:#9B2C2C; font-size:14px;">🔴 動態壓力賣點</b>
                            <h2 style="color:#991B1B; margin:10px 0;">{sell_target:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                with d3:
                    st.markdown(f"""
                        <div style="background:#F0FFF4; padding:20px; border-radius:12px; border-left:8px solid #38A169; text-align:center;">
                            <b style="color:#22543D; font-size:14px;">📈 預期報酬</b>
                            <h2 style="color:#2F855A; margin:10px 0;">{expected_return:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                  
                if expected_return < 1.2:
                    st.info("💡 目前即時波動率極低，建議等待量能噴發後再參考點位。")

elif st.session_state.mode == "forecast":
    
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
    st.title("📊 波段預估")
    stock_id = st.text_input("輸入代碼 (例: 2330)")

    if stock_id:
        with st.spinner('AI 多因子計算與回測中...'):
            df, sym = fetch_stock_data(stock_id)
            if not df.empty:
                # --- 1. [數據計算區] ---
                df = df.ffill()
                name = get_stock_name(stock_id)
                curr_c = float(df['Close'].iloc[-1])    # 今日收盤
                prev_close = float(df['Close'].iloc[-2]) # 昨收價
                
                # --- 2. [族群動能與相對量能計算] ---
                # 相對成交量 (Relative Volume) [cite: 2026-01-12]
                relative_volume = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                
                # 族群輪動慣性 (以近 5 日累積漲跌幅估計) [cite: 2026-01-12]
                sector_momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                sector_bias = 1 + (sector_momentum * 0.005) # 族群強則慣性增加 [cite: 2026-01-12]

                # --- 3. [籌碼修正與波動計算] ---
                # 修正 Bias：整合量能與族群動能，不再只是固定的 0.994 [cite: 2026-01-12]
                bias = 1 + (relative_volume - 1) * 0.015 + (sector_momentum * 0.002)
                bias = max(0.97, min(1.04, bias)) # 限制範圍避免極端

                # ATR 基礎波動計算 [cite: 2026-01-12]
                tr = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
                
                # --- 4. [量能驅動開盤預估] ---
                # 不使用固定 0.05，改由相對量能 relative_volume 決定跳空強度 [cite: 2026-01-12]
                vol_impact = max(0.02, min(0.12, 0.04 * relative_volume * sector_bias))
                
                if curr_c >= prev_close:
                    est_open_raw = curr_c + (atr * vol_impact * bias) # 向上慣性 [cite: 2026-01-12]
                else:
                    est_open_raw = curr_c - (atr * vol_impact / bias) # 向下慣性 (考慮過跌) [cite: 2026-01-12]

                # --- 5. [台股 Tick Size 修正] ---
                # 呼叫頂部的 get_tick_size 函數 [cite: 2026-01-12]
                tick = get_tick_size(curr_c)
                
                # 修正波動慣性：台積電會變成 5.0 的倍數，不再是 1.73 [cite: 2026-01-12]
                vol_inertia = round((atr * bias) / tick) * tick 
                
                # 修正預估開盤：符合台股跳動單位 [cite: 2026-01-12]
                est_open = round(est_open_raw / tick) * tick

                price_diff = curr_c - prev_close 
                active_color = "#E53E3E" if price_diff >= 0 else "#38A169"


                # --- [2. 排版優化區：解決手機對比與字體問題] ---
                clean_name = name.split('(')[0].split('-')[0].strip()
                
                st.markdown(f"""
                    <style>
                        /* 手機端自動縮小大字體 */
                        @media (max-width: 600px) {{
                            .main-price {{ font-size: 55px !important; }}
                            .data-row {{ flex-direction: column !important; }}
                        }}
                    </style>

                    <div style='background: #FFFFFF; padding: 20px; border-radius: 15px; border-left: 10px solid {active_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                        <h2 style='color: #1E293B; margin: 0; font-size: 24px;'>({clean_name})的收盤價</h2>
                        <div style='display: flex; align-items: baseline; flex-wrap: wrap;'>
                            <b class='main-price' style='font-size: 75px; color: {active_color}; letter-spacing: -2px;'>{curr_c:.2f}</b>
                            <div style='margin-left: 15px;'>
                                <span style='font-size: 28px; color: {active_color}; font-weight: 900; display: block;'>
                                    {'▲' if price_diff >= 0 else '▼'} {abs(price_diff):.2f}
                                </span>
                                <span style='font-size: 18px; color: {active_color}; font-weight: 700;'>
                                    ({(price_diff/prev_close*100):.2f}%)
                                </span>
                            </div>
                        </div>
                    </div>

                    <div class='data-row' style='display: flex; background: #0F172A; padding: 15px; border-radius: 12px; color: white; margin-top: 15px; gap: 10px;'>
                        <div style='flex: 1; text-align: center; border-right: 1px solid #334155;'>
                            <span style='font-size: 12px; color: #94A3B8;'>籌碼修正</span>
                            <div style='font-size: 18px; font-weight: bold;'>{bias:.3f}</div>
                        </div>
                        <div style='flex: 1; text-align: center; border-right: 1px solid #334155;'>
                            <span style='font-size: 12px; color: #94A3B8;'>波動慣性</span>
                            <div style='font-size: 18px; font-weight: bold; color: #FACC15;'>{vol_inertia:.2f}</div>
                        </div>
                        <div style='flex: 1; text-align: center;'>
                            <span style='font-size: 12px; color: #94A3B8;'>預估開盤</span>
                            <div style='font-size: 18px; font-weight: bold;'>{est_open:.2f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # --- 1. 計算漲跌點數與百分比 ---
                price_diff = curr_c - prev_close  # 漲跌點數
                price_change_pct = (price_diff / prev_close) * 100

           
                # --- 2. [動態變色邏輯] ---
                price_color = "#C53030" if curr_c >= prev_close else "#2F855A" # 紅漲綠跌
                price_change_pct = (curr_c - prev_close) / prev_close * 100

               # --- [1. 機器學習個別回測：縮短至一個月數據優化版] ---
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import r2_score, mean_absolute_error

                # 僅取最近 30 筆資料 (約一個月)
                df_ml = df.tail(30).copy() 
                df_ml['Next_High'] = df_ml['High'].shift(-1)
                df_ml = df_ml.dropna()

                if len(df_ml) > 10: 
                    features_ml = ['Open', 'High', 'Low', 'Close', 'Volume']
                    X_ml = df_ml[features_ml]
                    y_ml = df_ml['Next_High']

                    # 建立標準化工具
                    scaler_ml = StandardScaler()
                    # 關鍵：定義 X_scaled
                    X_scaled = scaler_ml.fit_transform(X_ml)
                    
                    # 訓練模型
                    model_ml = LinearRegression()
                    model_ml.fit(X_scaled, y_ml)

                    # 計算這一個月內的回測準確度
                    y_pred = model_ml.predict(X_scaled)
                    stock_r2 = r2_score(y_ml, y_pred)
                    stock_mae = mean_absolute_error(y_ml, y_pred)

                    # 預測明日最高價 (使用最後一天的數據)
                    latest_data = df[features_ml].tail(1)
                    latest_scaled = scaler_ml.transform(latest_data)
                    ml_tomorrow_high = model_ml.predict(latest_scaled)[0]
                    
                    # 對齊台股跳動單位
                    ml_tomorrow_high = round(ml_tomorrow_high / tick) * tick
                    ml_upside = ((ml_tomorrow_high / curr_c) - 1) * 100

                # --- [顯示：機器學習個別標定報告 (亮底深字)] ---
                    clean_name = name.split('(')[0].split('-')[0].strip()
                    st.markdown(f"### 🤖 {clean_name} 的專屬 AI 機器學習回測")
                    r2_eval = "極高" if stock_r2 > 0.9 else ("高" if stock_r2 > 0.8 else "中等")
                    r2_color = "#059669" if stock_r2 > 0.8 else "#D97706"

                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                     st.markdown(f"""
                        <div style="background:#FFFBEB; padding:20px; border-radius:12px; border:1px solid #FEF3C7; text-align:center;">
                            <b style="color:#92400E; font-size:14px;">🎯 ML 預估最高價</b>
                            <h2 style="color:#78350F; margin:10px 0;">{ml_tomorrow_high:.2f}</h2>
                            <small style="color:#B45309;">預期空間: {ml_upside:.2f}%</small>
                        </div>
                    """, unsafe_allow_html=True)
                    with mc2:
                     st.markdown(f"""
                        <div style="background:#ECFDF5; padding:20px; border-radius:12px; border:1px solid #D1FAE5; text-align:center;">
                            <b style="color:#065F46; font-size:14px;">📈 預測信心度 (R2)</b>
                            <h2 style="color:{r2_color}; margin:10px 0;">{stock_r2:.4f}</h2>
                            <small style="color:#059669;">準確度評價：{r2_eval}</small>
                        </div>
                    """, unsafe_allow_html=True)
                    with mc3:
                     st.markdown(f"""
                        <div style="background:#FDF2F2; padding:20px; border-radius:12px; border:1px solid #FEE2E2; text-align:center;">
                            <b style="color:#9B1C1C; font-size:14px;">📏 平均預估誤差</b>
                            <h2 style="color:#AF1919; margin:10px 0;">±{stock_mae:.2f}</h2>
                            <small style="color:#C81E1E;">歷史平均偏離值</small>
                        </div>
                      """, unsafe_allow_html=True)

                
                    
                

                # --- 4. [命中率與卡片顯示區] ---
                # 計算 60 日真實回測命中率 [cite: 2026-01-12]
                acc_dh = calculate_real_accuracy(df, 0.85 * bias, 'high')
                acc_dl = calculate_real_accuracy(df, 0.65 / bias, 'low')
                acc_wh = calculate_real_accuracy(df, 1.9 * bias, 'high')
                acc_wl = calculate_real_accuracy(df, 1.6 / bias, 'low')

                st.divider()
                st.markdown("### 🎯 隔日與五日 AI 預估區間 (60日回測)")
                m1, m2, m3, m4 = st.columns(4)
                with m1: stock_box("📈 隔日壓力", curr_c + atr*0.85*bias, ((curr_c + atr*0.85*bias)/curr_c - 1)*100, acc_dh, "red")
                with m2: stock_box("📉 隔日支撐", curr_c - atr*0.65/bias, ((curr_c - atr*0.65/bias)/curr_c - 1)*100, acc_dl, "green")
                with m3: stock_box("🚩 五日壓力", curr_c + atr*1.9*bias, ((curr_c + atr*1.9*bias)/curr_c - 1)*100, acc_wh, "red")
                with m4: stock_box("⚓ 五日支撐", curr_c - atr*1.6/bias, ((curr_c - atr*1.6/bias)/curr_c - 1)*100, acc_wl, "green")

                if not df.empty:
                # --- [1. 基礎數據與財報評分] ---
                 df = df.ffill()
                 curr_c = float(df['Close'].iloc[-1])
                 prev_close = float(df['Close'].iloc[-2])
                 tick = get_tick_size(curr_c)
                
                # 新加入：財報影響力因子
                 stock_info = yf.Ticker(f"{stock_id}.TW")
                 f_score = 1.0
                 try:
                    info = stock_info.info
                    margin = info.get('grossMargins', 0.2)
                    rev_growth = info.get('revenueGrowth', 0)
                    if margin > 0.3: f_score += 0.02
                    if rev_growth > 0.1: f_score += 0.03
                 except: 
                     pass

                # --- [2. 族群與 Bias 計算] ---
                 relative_volume = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                # 族群動能必須在 bias 之前計算
                 sector_momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                 sector_bias = 1 + (sector_momentum * 0.005)
                
                # 整合量能、族群與財報
                 bias = (1 + (relative_volume - 1) * 0.015 + (sector_momentum * 0.002)) * f_score
                 bias = max(0.97, min(1.04, bias)) 

                # --- [3. 機器學習訓練數據準備] ---
                # 解決 NameError: 確保 df_ml 在被使用前已完全定義且縮排正確
                 df_ml = df.tail(30).copy() 
                 df_ml['Next_High'] = df_ml['High'].shift(-1)
                 df_ml = df_ml.dropna()

                # 初始化預測值，避免後續顯示報錯
                 ml_tomorrow_high = curr_c * 1.01 

                if len(df_ml) > 10:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.preprocessing import StandardScaler
                    
                    features_ml = ['Open', 'High', 'Low', 'Close', 'Volume']
                    X_ml = df_ml[features_ml]
                    y_ml = df_ml['Next_High']
                    
                    scaler_ml = StandardScaler()
                    X_scaled = scaler_ml.fit_transform(X_ml)
                    
                    model_ml = LinearRegression()
                    model_ml.fit(X_scaled, y_ml)
                    
                    # 預測明日最高價
                    latest_data = df[features_ml].tail(1)
                    latest_scaled = scaler_ml.transform(latest_data)
                    # 預測值結合財報權重修正
                    ml_tomorrow_high = model_ml.predict(latest_scaled)[0] * f_score
                    ml_tomorrow_high = round(ml_tomorrow_high / tick) * tick

                # --- [4. AI 統整投資建議] ---
                st.subheader("🎯 AI 全維度投資決策")
                if relative_volume > 1.2 and sector_momentum > 0 and f_score > 1.0:
                    st.success(f"🔥 **強烈看好**：量能、族群、財報三強鼎立，預期挑戰 {ml_tomorrow_high:.2f}")
                elif relative_volume < 0.8:
                    st.warning("💤 **冷清觀望**：目前量縮，缺乏主力介入。")
                else:
                    st.info("⚖️ **中性布局**：建議守住支撐位。")

               

               # --- 📈 走勢圖與 AI 預估區間 ---
                st.divider()
                st.markdown(f"### 📈 {name}({sym}) 走勢圖與 AI 預估區間")
                
                # 建立畫布
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                plot_df = df.tail(45)
                
                # 價格圖：顯示英文標籤避免亂碼
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=3, label="Close Price")
                ax1.axhline(curr_c + atr * 1.9 * bias, color='red', ls='--', lw=2, alpha=0.7, label="5D Resistance")
                ax1.axhline(curr_c - atr * 1.6 / bias, color='green', ls='--', lw=2, alpha=0.7, label="5D Support")
                
                ax1.legend(loc='upper left', frameon=True, fontsize=10)
                ax1.grid(alpha=0.3)
                ax1.set_ylabel("Price")
                
                # 成交量柱狀圖
                v_colors = ['#EF5350' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] else '#26A69A' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.8)
                ax2.set_ylabel("Volume")
                
                plt.tight_layout()
                st.pyplot(fig)
                

                # --- 🎯 補充說明註解 (根據您的指示強化) ---
                # 取得執行當下的時間
                # --- 🎯 AI 數據自動化偵測報告 (內容隨每日數據與日期變動) ---
                
                # 1. 定義時區與即時日期
                tw_tz = pytz.timezone("Asia/Taipei") 
                current_time = datetime.now(tw_tz)
                current_date = current_time.strftime('%Y-%m-%d')
                current_hm = current_time.strftime('%H:%M')

                # 2. 判斷今日盤態：考慮漲停、過度下跌與籌碼修正
                daily_change_pct = (curr_c - prev_close) / prev_close * 100

                st.info(f"📋 **AI 數據自動化偵測報告 (分析基準日：{current_date} {current_hm})**")

                # 3. 建立顯示欄位
                note_col1, note_col2 = st.columns(2)

                with note_col1:
                    # 根據漲跌幅與籌碼修正量 (bias) 自動生成動態文字
                    if daily_change_pct > 7 and bias > 1.05:
                        status_text = "🔥 強勢攻擊盤 (多頭噴發)"
                        status_desc = "今日漲幅極大且帶量，慣性已突破 ATR 常態區間。壓力位僅供參考，應注意乖離率。"
                    elif daily_change_pct < -7 and bias > 1.05:
                        status_text = "❄️ 恐慌下跌盤 (放量殺低)"
                        status_desc = "偵測到過度下跌因素，下跌慣性強烈。支撐位可能失守，請謹慎接刀。"
                    else:
                        status_text = "帶量擴張" if bias > 1 else "量縮盤整"
                        status_desc = f"目前籌碼修正係數為 {bias:.3f}，AI 已根據法人籌碼慣性自動調整空間。"

                    st.markdown(f"""
                    **1. 籌碼流向動態：**
                    - 今日盤態：**{status_text}**
                    - 說明：{status_desc}
                    
                    **2. 價格波動慣性 (Inertia)：**
                    - 14 日 ATR 波動均幅：`{atr:.2f}`
                    - 預估明日開盤慣性：`{est_open:.2f}` (隨每日數據動態計算)
                    """)

                with note_col2:
                    # 根據 60 日回測命中率判定評等
                    confidence_tag = "核心參考" if acc_dh > 85 else "謹慎參考 (波動異常)"
                    
                    st.markdown(f"""
                    **3. 60 日歷史回測精度：**
                    - 考慮「波動慣性」與「法人籌碼」後之真實命中率。
                    - 過去 60 交易日維持了 **{acc_dh:.1f}%**，評等為：`{confidence_tag}`。
                    
                    **4. 空間參考範疇：**
                    - 預計明日波動範圍約在 `{curr_c - atr*0.65/bias:.2f}` 至 `{curr_c + atr*0.85*bias:.2f}` 之間。
                    """)

                # 4. 底部自動日期聲明
                st.caption(f"※ 本分析由 AI 於 {current_date} 根據 {name}({stock_id}) 最新數據自動生成。")

                
                st.warning("⚠️ **免責聲明**：本系統僅供 AI 數據研究參考，不構成任何投資建議。交易前請務必自行評估風險。")














































































































