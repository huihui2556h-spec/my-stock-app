import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import urllib3
import os
from datetime import datetime, time, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib

# --- [全域初始化與網頁設定] ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股 AI 交易助手 Pro - 全功能回測版", layout="wide", page_icon="💹")

# 設定時區
tw_tz = pytz.timezone("Asia/Taipei")

# --- [字體防亂碼全英設定] ---
matplotlib.rcParams['axes.unicode_minus'] = False 

# 預測紀錄檔路徑
DB_FILE = "prediction_history.csv"

# --- [全域變數與金鑰] ---
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0wNSAxODozOToxOSIsInVzZXJfaWQiOiJhYXJvbjA3IiwiZW1haWwiOiJodWlodWkyNTU2aEBnbWFpbC5jb20iLCJpcCI6IjEuMTcwLjkwLjIyNSJ9.n-uv7ODTCIAjl0mffN2_rsIvqwLRWB3rVFCBd7jG0bE"

name_map = {
    "PCB-CCL": "PCB-材料 (CCL/銅箔)", "PCB-Substrate": "PCB-載板 (ABF/BT)", "PCB-Assembly": "PCB-組裝加工 (硬板/HDI)",
    "Memory-Fab": "記憶體-原廠/代工", "Memory-Module": "記憶體-模組廠", "Memory-Controller": "記憶體-控制 IC", "Memory-DDR5": "記憶體-DDR5/高速傳輸",
    "Semi-Equip": "半導體-設備/CoWoS", "Semi-OSAT": "半導體-封測 (先進封裝/測試)", "AI-ASIC": "AI 特用晶片 (矽智財/ASIC)",
    "AI-Case": "AI 伺服器 (機殼/滑軌)", "AI-Cooling": "AI 伺服器 (散熱/水冷)", "AI-ODM": "AI 伺服器 (ODM 代工)",
    "CPO-Silicon": "矽光子 (CPO/光通訊)", "Satellite-LEO": "低軌衛星 (航太/地面站)", "Display-Panel": "面板-驅動 IC/面板廠",
    "Passive-Comp": "被動元件 (MLCC/電阻)", "Optical-Lens": "光學鏡頭 (手機/車載)", "Auto-EV": "車用電子 (電動車/二極體)",
    "Power-Grid": "重電/電力 (政策股)", "Shipping": "航運 (貨櫃/散裝)"
}

INDUSTRY_CHAINS_EN = {
    "PCB-CCL": ["6213.TW", "2383.TW", "6274.TW", "8358.TWO", "2367.TW"],
    "PCB-Substrate": ["8046.TW", "3037.TW", "3189.TW", "6667.TW"],
    "PCB-Assembly": ["2367.TW", "2313.TW", "2368.TW", "4958.TW", "3044.TW"],
    "Memory-Fab": ["2344.TW", "2337.TW", "2408.TW", "3006.TW"],
    "Memory-Module": ["3260.TWO", "8299.TW", "2451.TW", "3264.TWO", "3546.TW"],
    "Memory-Controller": ["8299.TW", "4966.TW", "6233.TWO", "6104.TW"],
    "Memory-DDR5": ["6138.TW", "6213.TW", "8299.TW", "3260.TWO", "6515.TW"],
    "Semi-Equip": ["3131.TWO", "3583.TW", "1560.TW", "6187.TWO", "2467.TW", "3680.TW"],
    "Semi-OSAT": ["2311.TW", "3711.TW", "2449.TW", "6147.TWO", "6239.TW"],
    "AI-ASIC": ["3661.TW", "3443.TW", "6643.TW", "3035.TW", "8227.TW"],
    "AI-ODM": ["2382.TW", "2317.TW", "3231.TW", "6669.TW", "2356.TW", "2376.TW"],
    "AI-Cooling": ["3017.TW", "3324.TW", "2421.TW", "6230.TW", "3483.TW"],
    "AI-Case": ["8210.TW", "2059.TW", "6803.TW", "3693.TW", "3013.TW"],
    "CPO-Silicon": ["3363.TWO", "4979.TWO", "3081.TWO", "6451.TW", "3450.TW"],
    "Satellite-LEO": ["2313.TW", "3491.TWO", "2314.TW", "3380.TW", "6285.TW"],
    "Display-Panel": ["2409.TW", "3481.TW", "6116.TW", "3034.TW", "3545.TW"],
    "Passive-Comp": ["2327.TW", "2492.TW", "6173.TWO", "6127.TWO", "2456.TW"],
    "Optical-Lens": ["3008.TW", "3406.TW", "3362.TW", "3504.TW", "3441.TWO"],
    "Shipping": ["2603.TW", "2609.TW", "2615.TW", "2606.TW", "2637.TW"],
    "Power-Grid": ["1513.TW", "1503.TW", "1519.TW", "1514.TW", "1504.TW"],
    "Auto-EV": ["2317.TW", "2481.TW", "5425.TWO", "3675.TW", "2351.TW"]
}

# --- [工具函數區] ---
def get_tick_size(price):
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0

def get_stock_name(stock_id):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        html = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5).text
        name = re.search(r'<title>(.*?) \(', html).group(1)
        return name.split('-')[0].strip()
    except:
        return f"台股 {stock_id}"

@st.cache_data(ttl=60)
def fetch_stock_data(stock_id, period="150d"):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        try:
            df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.strip() for c in df.columns]
                df = df.dropna(subset=['Close'])
                return df, symbol
        except:
            continue
    return pd.DataFrame(), None

# --- [歷史預測資料庫核心邏輯] ---
def save_prediction(stock_id, stock_name, current_price, pred_low, pred_high):
    today_str = datetime.now(tw_tz).strftime("%Y-%m-%d")
    new_data = pd.DataFrame([{
        "prediction_date": today_str,
        "stock_id": stock_id,
        "stock_name": stock_name,
        "base_price": round(current_price, 2),
        "pred_next_low": round(pred_low, 2),
        "pred_next_high": round(pred_high, 2),
        "actual_next_low": np.nan,
        "actual_next_high": np.nan,
        "is_hit": "Pending"
    }])
    
    if os.path.exists(DB_FILE):
        try:
            df_db = pd.read_csv(DB_FILE)
            df_db = df_db[~((df_db['prediction_date'] == today_str) & (df_db['stock_id'] == stock_id))]
            df_db = pd.concat([df_db, new_data], ignore_index=True)
        except:
            df_db = new_data
    else:
        df_db = new_data
        
    df_db.to_csv(DB_FILE, index=False, encoding="utf-8-sig")

def update_and_calculate_accuracy():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame(), 0.0
    try:
        df_db = pd.read_csv(DB_FILE)
    except:
        return pd.DataFrame(), 0.0
        
    updated = False
    for idx, row in df_db.iterrows():
        if row['is_hit'] == "Pending" or pd.isna(row['actual_next_low']):
            pred_dt = datetime.strptime(row['prediction_date'], "%Y-%m-%d")
            if pred_dt.date() < datetime.now(tw_tz).date():
                hist_df, _ = fetch_stock_data(row['stock_id'], period="10d")
                if not hist_df.empty:
                    post_df = hist_df[hist_df.index > pd.to_datetime(row['prediction_date'])]
                    if not post_df.empty:
                        act_low = float(post_df['Low'].iloc[0])
                        act_high = float(post_df['High'].iloc[0])
                        df_db.at[idx, 'actual_next_low'] = round(act_low, 2)
                        df_db.at[idx, 'actual_next_high'] = round(act_high, 2)
                        
                        if (act_high >= row['pred_next_low']) and (act_low <= row['pred_next_high']):
                            df_db.at[idx, 'is_hit'] = "Hit (成功)"
                        else:
                            df_db.at[idx, 'is_hit'] = "Miss (未命中)"
                        updated = True
                        
    if updated:
        df_db.to_csv(DB_FILE, index=False, encoding="utf-8-sig")
        
    hit_rows = df_db[df_db['is_hit'] == "Hit (成功)"]
    closed_rows = df_db[df_db['is_hit'].isin(["Hit (成功)", "Miss (未命中)"])]
    accuracy = (len(hit_rows) / len(closed_rows) * 100) if not closed_rows.empty else 0.0
    return df_db, accuracy

# --- [Session State 初始化] ---
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

# =========================================================
# 主程式路由分流
# =========================================================

# --- 【HOME：首頁導覽】 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策與紀律回測系統 Pro")
    st.markdown("歡迎進入整合式全方位決策系統，請點選下方功能進入專屬儀表板：")
    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 盤中即時量價 (當沖監控)", use_container_width=True): st.session_state.mode = "realtime"; st.rerun()
        if st.button("🔮 隔日區間預估 與 紀律存檔", use_container_width=True): st.session_state.mode = "forecast"; st.rerun()
    with col_b:
        if st.button("💎 類群輪動大戶預警", use_container_width=True): st.session_state.mode = "sector"; st.rerun()
        if st.button("🆘 拯救套牢診斷艙", use_container_width=True): st.session_state.mode = "rescue"; st.rerun()
        
    st.divider()
    if st.button("📊 進入 每日真實準確率 回測中心看板", use_container_width=True):
        st.session_state.mode = "backtest"
        st.rerun()

# --- 【REALTIME：盤中即時量價頁面】 ---
elif st.session_state.mode == "realtime":
    st.title("⚡ 盤中即時量價（當沖監控）")
    if st.button("⬅️ 返回首頁"): st.session_state.mode = "home"; st.rerun()
    st.divider()
    now = datetime.now(tw_tz)
    is_market_open = now.weekday() < 5 and (time(9, 0) <= now.time() <= time(13, 30))
    stock_id = st.text_input("輸入股票代碼（如：2330）")

    if stock_id:
        df, sym = fetch_stock_data(stock_id, period="60d")
        if df.empty:
            st.error("❌ 查無資料，請檢查代碼是否正確。")
        else:
            df = df.ffill()
            name = get_stock_name(stock_id)
            curr_price = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            price_diff = curr_price - prev_close
            active_color = "#E53E3E" if price_diff >= 0 else "#38A169"

            if not is_market_open:
                st.warning(f"🕒 【目前非交易時段】系統暫停動態當沖演算。現在時間：{now.strftime('%H:%M')}。")
            else:
                st.success(f"🟢 【盤中 AI 動態監控中】數據隨量價即時校正")

            recent_std = df['Close'].tail(15).std()
            avg_vol = df['Volume'].tail(10).mean()
            instant_vol_factor = df['Volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1.0

            st.markdown(f"""
                <div style='background: #FFFFFF; padding: 25px; border-radius: 18px; border-left: 12px solid {active_color}; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;'>
                    <div style='color: #0F172A; font-size: 28px; font-weight: 800;'>{name} ({sym})</div>
                    <div style='display: flex; align-items: baseline; flex-wrap: wrap; margin-top:10px;'>
                        <b style='font-size: 70px; color: {active_color}; line-height: 1;'>{curr_price:.2f}</b>
                        <div style='margin-left: 15px;'>
                            <span style='font-size: 28px; color: {active_color}; font-weight: 900; display: block;'>{'▲' if price_diff >= 0 else '▼'} {abs(price_diff):.2f}</span>
                            <span style='font-size: 18px; color: {active_color}; font-weight: 700;'>({(price_diff/prev_close*100):.2f}%)</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            stability_index = df['Close'].tail(5).std() / recent_std
            confidence_shield = max(1.0, min(2.0, stability_index))
            vol_expansion = np.sqrt(instant_vol_factor) 
            
            dynamic_offset_low = recent_std * (confidence_shield / vol_expansion)
            dynamic_offset_high = recent_std * (vol_expansion * confidence_shield)
            
            tick = get_tick_size(curr_price)
            buy_point = round((curr_price - dynamic_offset_low) / tick) * tick
            sell_target = round((curr_price + dynamic_offset_high) / tick) * tick
            expected_return = (sell_target - buy_point) / buy_point * 100

            st.subheader("🎯 當沖 AI 動態演算建議")
            d1, d2, d3 = st.columns(3)
            with d1: st.markdown(f"<div style='background:#F0F9FF; padding:20px; border-radius:12px; text-align:center;'><b>🔹 動態支撐買點</b><h2>{buy_point:.2f}</h2></div>", unsafe_allow_html=True)
            with d2: st.markdown(f"<div style='background:#FFF5F5; padding:20px; border-radius:12px; text-align:center;'><b>🔴 動態壓力賣點</b><h2>{sell_target:.2f}</h2></div>", unsafe_allow_html=True)
            with d3: st.markdown(f"<div style='background:#F0FFF4; padding:20px; border-radius:12px; text-align:center;'><b>📈 預期報酬範圍</b><h2>{expected_return:.2f}%</h2></div>", unsafe_allow_html=True)

# --- 【FORECAST：隔日區間預估 與 未來中長期波段 頁面】 ---
elif st.session_state.mode == "forecast":
    if st.button("⬅️ 返回首頁"): st.session_state.mode = "home"; st.rerun()
        
    st.title("🔮 波段未來目標價預估 與 隔日極端區間")
    stock_id = st.text_input("請輸入股票代碼（如：2330, 2408, 8358）")

    if stock_id:
        with st.spinner('AI 正在計算波段擴展滿足點與隔日波動區間...'):
            df, sym = fetch_stock_data(stock_id, period="150d")
            
            if not df.empty:
                df = df.ffill()
                name = get_stock_name(stock_id)
                curr_c = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                price_diff = curr_c - prev_close
                active_color = "#E53E3E" if price_diff >= 0 else "#38A169"
                tick = get_tick_size(curr_c)
                
                # ------【1. 隔日短線波動演算 (ATR模型)】------
                df['TR'] = np.maximum(df['High'] - df['Low'], 
                                      np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                                 abs(df['Low'] - df['Close'].shift(1))))
                df['ATR_5'] = df['TR'].rolling(window=5).mean()
                current_atr = df['ATR_5'].iloc[-1] if not pd.isna(df['ATR_5'].iloc[-1]) else (df['High'].iloc[-5:] - df['Low'].iloc[-5:]).mean()
                
                final_next_low = round((curr_c - current_atr * 1.1) / tick) * tick
                final_next_high = round((curr_c + current_atr * 1.1) / tick) * tick

                # ------【2. 中長期波段目標價 (黃金分割擴展模型)】------
                df['BB_MA'] = df['Close'].rolling(window=20).mean()
                df['BB_STD'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_MA'] + (2 * df['BB_STD'])
                df['BB_Lower'] = df['BB_MA'] - (2 * df['BB_STD'])

                backtest_df = df.tail(100)
                p_min_idx = backtest_df['Close'].idxmin()
                p_max_idx = backtest_df['Close'].idxmax()
                wave_low = backtest_df.loc[p_min_idx, 'Close']
                wave_high = backtest_df.loc[p_max_idx, 'Close']
                wave_height = wave_high - wave_low

                if p_min_idx < p_max_idx:
                    target_low = wave_low + (wave_height * 1.382)
                    target_high = wave_low + (wave_height * 1.618)
                    trend_status = "📈 多頭結構：主升段推升，突破前高後之未來擴展波段目標。"
                else:
                    target_low = wave_low + (wave_height * 0.382)
                    target_high = wave_low + (wave_height * 0.618)
                    trend_status = "📉 修正結構：波段築底階段，未來中線強彈之壓力波段目標區間。"

                final_target_min = round(target_low / tick) * tick
                final_target_max = round(target_high / tick) * tick
                if final_target_min == final_target_max: final_target_max += tick

                # 歷史軌道覆蓋率
                hit_counts = sum(1 for i in range(len(df) - 100, len(df)) if df['BB_Lower'].iloc[i] <= df['Close'].iloc[i] <= df['BB_Upper'].iloc[i])
                bb_accuracy = (hit_counts / 100) * 100

                # --- 介面呈現 ---
                st.markdown(f"""
                    <div style='background: #FFFFFF; padding: 20px; border-radius: 15px; border-left: 10px solid {active_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                        <h2 style='color: #1E293B; margin: 0; font-size: 22px;'>{name} ({stock_id}) 今日收盤：{curr_c:.2f} ({'▲' if price_diff >= 0 else '▼'}{abs(price_diff):.2f})</h2>
                    </div>
                """, unsafe_allow_html=True)

                # 看板 1：隔日可能觸及區間
                st.markdown("### ⚡ 短線防守：隔日預期震盪範圍")
                cc1, cc2 = st.columns(2)
                with cc1: st.markdown(f"<div style='background:#FFF5F5; padding:15px; border-radius:8px; text-align:center;'><span style='color:#C53030;'>📈 隔日預估可能最高觸及</span><h2>{final_next_high:.2f}</h2></div>", unsafe_allow_html=True)
                with cc2: st.markdown(f"<div style='background:#F0FFF4; padding:15px; border-radius:8px; text-align:center;'><span style='color:#2F855A;'>📉 隔日預估可能最低觸及</span><h2>{final_next_low:.2f}</h2></div>", unsafe_allow_html=True)

                # 看板 2：未來中長期波段目標
                st.markdown("### 🔮 中長線波段：未來擴展目標價區間 (如 100 → 140~150 概念)")
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); padding: 25px; border-radius: 16px; color: white; box-shadow: 0 6px 15px rgba(0,0,0,0.15);">
                        <div style="font-size: 48px; font-weight: 800; color: #F59E0B;">{final_target_min:.2f} <span style="font-size:22px; color:#93C5FD;">至</span> {final_target_max:.2f}</div>
                        <div style="font-size: 14px; margin-top: 10px; color:#E2E8F0;">🎯 趨勢狀態：{trend_status}</div>
                    </div>
                """, unsafe_allow_html=True)

                # 每日紀律存檔按鈕
                st.divider()
                st.markdown("### 💾 執行每日紀律預測存檔")
                if st.button("📥 記錄今日預測資料（納入勝率計算）", use_container_width=True):
                    save_prediction(stock_id, name, curr_c, final_next_low, final_next_high)
                    st.success(f"🎉 成功存檔！已將 {name} 今日預測範圍記錄至資料庫。")

                # 繪圖
                st.divider()
                st.subheader("📈 技術指標與趨勢波浪軌道追蹤 (全英圖例)")
                plot_df = df.tail(100)
                fig, ax = plt.subplots(figsize=(11, 4.5))
                ax.plot(plot_df.index, plot_df['Close'], label='Close Price', color='#1E293B', linewidth=2)
                ax.plot(plot_df.index, plot_df['BB_MA'], label='BB Middle (20 MA)', color='#3B82F6', linestyle='--')
                ax.plot(plot_df.index, plot_df['BB_Upper'], label='BB Upper (2 Std)', color='#EF4444', alpha=0.6)
                ax.plot(plot_df.index, plot_df['BB_Lower'], label='BB Lower (2 Std)', color='#10B981', alpha=0.6)
                ax.scatter(p_min_idx, wave_low, color='#10B981', s=120, marker='^', label='100D Wave Low Base')
                ax.scatter(p_max_idx, wave_high, color='#EF4444', s=120, marker='v', label='100D Wave High Peak')
                ax.set_title(f"{stock_id} Wave Trend Dashboard", fontsize=10, fontweight='bold')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, linestyle=':', alpha=0.5)
                st.pyplot(fig)
            else:
                st.error("❌ 無法取得該股票歷史資料。")

# --- 【BACKTEST：每日真實準確率回測看板】 ---
elif st.session_state.mode == "backtest":
    if st.button("⬅️ 返回首頁"): st.session_state.mode = "home"; st.rerun()
    st.title("📊 每日預測紀律與真實準確率回測中心")
    
    with st.spinner('正在對齊歷史數據並更新勝率...'):
        df_db, total_acc = update_and_calculate_accuracy()
        if df_db.empty:
            st.warning("📭 目前無歷史預測紀錄！請先至「隔日區間預估」儲存您的觀察標的。")
        else:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E1B4B 0%, #431407 100%); padding: 25px; border-radius: 16px; color: white; margin-bottom: 25px;">
                    <span style="font-size: 14px; color: #FED7AA; font-weight: bold;">📊 AI 模型實戰真實勝率 (隔日觸及成功率)</span>
                    <div style="font-size: 60px; font-weight: 900; color: #F97316;">{total_acc:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            df_display = df_db.copy().sort_values(by="prediction_date", ascending=False)
            df_display.columns = ["預測日期", "股票代碼", "股票名稱", "預測基準價", "預估隔日最低", "預估隔日最高", "實際隔日最低", "實際隔日最高", "開獎結果"]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- 【SECTOR：類群輪動預警頁面】 ---
elif st.session_state.mode == "sector":
    st.title("💎 類群輪動預警儀表板")
    if st.button("⬅️ 返回首頁"): st.session_state.mode = "home"; st.rerun()
    st.divider()
    
    with st.spinner('AI 正在計算各細分產業鏈大戶資金流入強度...'):
        flow_report = []
        for en_id, tickers in INDUSTRY_CHAINS_EN.items():
            try:
                data = yf.download(tickers, period="10d", progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1).mean() * 100
                    vol_ratio = data['Volume'].iloc[-1].sum() / data['Volume'].tail(5).mean().sum()
                    flow_report.append({"ID": en_id, "漲跌%": ret, "資金流入": vol_ratio})
            except: continue
        
        df_flow = pd.DataFrame(flow_report)
        if not df_flow.empty:
            buy_candidates = df_flow[(df_flow['資金流入'] > 1.2) & (df_flow['漲跌%'] > -0.5)]
            st.subheader("🎯 當前強勢主流族群")
            if not buy_candidates.empty:
                best_sector_id = buy_candidates.sort_values(by="資金流入", ascending=False).iloc[0]['ID']
                st.success(f"🚀 **【大戶強烈聚焦關注】：{name_map[best_sector_id]}**")
                strong_tickers = INDUSTRY_CHAINS_EN.get(best_sector_id, [])
                if strong_tickers:
                    s_cols = st.columns(len(strong_tickers))
                    for idx, ticker in enumerate(strong_tickers):
                        try:
                            s_name = get_stock_name(ticker.split('.')[0]).replace("走勢圖", "").strip()
                            s_h = yf.Ticker(ticker).history(period="2d")
                            if len(s_h) >= 2:
                                s_ret = (s_h['Close'].iloc[-1] / s_h['Close'].iloc[-2] - 1) * 100
                                with s_cols[idx]: st.metric(label=s_name, value=f"{s_h['Close'].iloc[-1]:.2f}", delta=f"{s_ret:.2f}%")
                        except: continue
            
            st.divider()
            df_display = df_flow.copy()
            df_display['產業名稱'] = df_display['ID'].map(name_map)
            st.dataframe(df_display[['產業名稱', '漲跌%', '資金流入']].sort_values(by='資金流入', ascending=False), use_container_width=True, hide_index=True)

# --- 【RESCUE：拯救套牢診斷頁面】 ---
elif st.session_state.mode == "rescue":
    st.title("🆘 拯救套牢診斷艙")
    if st.button("⬅️ 返回首頁"): st.session_state.mode = "home"; st.rerun()
    st.divider()
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        r_id = st.text_input("請輸入套牢股票代碼：")
        r_cost = st.number_input("您的買進持股成本價：", min_value=0.0, step=0.1)
    with col_r2:
        r_volume = st.number_input("持有張數 (張)：", min_value=0, step=1)
        
    if r_id and r_cost > 0 and r_volume > 0:
        df, sym = fetch_stock_data(r_id, period="100d")
        if not df.empty:
            df = df.ffill(); curr_p = float(df['Close'].iloc[-1]); loss_pct = ((curr_p - r_cost) / r_cost) * 100
            st.subheader("📊 庫存損益現狀診斷")
            if loss_pct >= 0: st.success(f"🎉 帳面目前為獲利狀態！報酬率：+{loss_pct:.2f}%")
            else:
                st.error(f"❌ 目前處於套牢狀態。報酬率：{loss_pct:.2f}% (現價：{curr_p:.2f})")
                support_p = df['Low'].tail(20).min(); resistance_p = df['High'].tail(20).max()
                st.markdown(f"> **⚠️ 技術面提示**：支撐位 <b style='color:#28A745;'>{support_p:.2f}</b> ｜ 解套壓力位 <b style='color:#DC3545;'>{resistance_p:.2f}</b>", unsafe_allow_html=True)
                
                st.divider()
                add_shares = st.slider("預計加碼買進張數 (張)：", min_value=1, max_value=r_volume * 3, value=r_volume)
                new_cost = ((r_cost * r_volume) + (curr_p * add_shares)) / (r_volume + add_shares)
                new_loss_pct = ((curr_p - new_cost) / new_cost) * 100
                
                c_m1, c_m2 = st.columns(2)
                with c_m1: st.metric(label="攤平後平均成本價", value=f"{new_cost:.2f}", delta=f"成本降低 {r_cost - new_cost:.2f}")
                with c_m2: st.metric(label="攤平後預估新報酬率", value=f"{new_loss_pct:.2f}%", delta=f"風險縮減 {abs(loss_pct) - abs(new_loss_pct):.2f}%")
