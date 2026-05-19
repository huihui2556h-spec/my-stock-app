import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import re
import urllib3
from datetime import datetime, time, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- [全域初始化與網頁設定] ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股 AI 交易助手 Pro", layout="wide", page_icon="💹")

# 設定時區
tw_tz = pytz.timezone("Asia/Taipei")

# --- [字體防亂碼設定] ---
def set_mpl_font():
    # 優先使用英文標籤搭配乾淨樣式，並嘗試載入常見系統中文字體作為備援
    fonts = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    for f in fonts:
        try:
            matplotlib.rc('font', family=f)
            plt.figure()
            plt.close()
            break
        except:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False 

set_mpl_font()

# --- [全域變數與金鑰] ---
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0wNSAxODozOToxOSIsInVzZXJfaWQiOiJhYXJvbjA3IiwiZW1haWwiOiJodWlodWkyNTU2aEBnbWFpbC5jb20iLCJpcCI6IjEuMTcwLjkwLjIyNSJ9.n-uv7ODTCIAjl0mffN2_rsIvqwLRWB3rVFCBd7jG0bE"

# 產業鏈名稱對照表
name_map = {
    "PCB-CCL": "PCB-材料 (CCL/銅箔)",
    "PCB-Substrate": "PCB-載板 (ABF/BT)",
    "PCB-Assembly": "PCB-組裝加工 (硬板/HDI)",
    "Memory-Fab": "記憶體-原廠/代工",
    "Memory-Module": "記憶體-模組廠",
    "Memory-Controller": "記憶體-控制 IC",
    "Memory-DDR5": "記憶體-DDR5/高速傳輸",
    "Semi-Equip": "半導體-設備/CoWoS",
    "Semi-OSAT": "半導體-封測 (先進封裝/測試)",
    "AI-ASIC": "AI 特用晶片 (矽智財/ASIC)",
    "AI-Case": "AI 伺服器 (機殼/滑軌)",
    "AI-Cooling": "AI 伺服器 (散熱/水冷)",
    "AI-ODM": "AI 伺服器 (ODM 代工)",
    "CPO-Silicon": "矽光子 (CPO/光通訊)",
    "Satellite-LEO": "低軌衛星 (航太/地面站)",
    "Display-Panel": "面板-驅動 IC/面板廠",
    "Passive-Comp": "被動元件 (MLCC/電阻)",
    "Optical-Lens": "光學鏡頭 (手機/車載)",
    "Auto-EV": "車用電子 (電動車/二極體)",
    "Power-Grid": "重電/電力 (政策股)",
    "Shipping": "航運 (貨櫃/散裝)"
}

# 產業鏈成分股英文 ID 定義
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
    """計算台股各檔位升降單位"""
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0

def get_stock_name(stock_id):
    """從 Yahoo 財經網頁動態爬取股票名稱"""
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        html = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5).text
        name = re.search(r'<title>(.*?) \(', html).group(1)
        return name.split('-')[0].strip()
    except:
        return f"台股 {stock_id}"

@st.cache_data(ttl=60)
def fetch_stock_data(stock_id, period="120d"):
    """安全抓取 yfinance 數據並拍平 MultiIndex 結構"""
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
            }
        except:
            continue
    return pd.DataFrame(), None

def fetch_finmind_chips(stock_id, token=FINMIND_TOKEN):
    """獲取 FinMind 法人籌碼面因子，失敗則自動啟動 yfinance 備援"""
    default_res = (1.0, 0.0, 0.0, 0.0, 0.0, "備援演算啟動")
    pure_id = str(stock_id).split('.')[0]
    id_variants = [pure_id, f"{pure_id}.TW"]
    
    for target_id in id_variants:
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            start_dt = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            params = {
                "dataset": "InstitutionalInvestorsBuySell",
                "data_id": target_id,
                "start_date": start_dt,
                "token": token
            }
            resp = requests.get(url, params=params, timeout=10, verify=False)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                if data:
                    df = pd.DataFrame(data)
                    df = df[(df['buy'] != 0) | (df['sell'] != 0)]
                    if not df.empty:
                        last_date = df['date'].max()
                        today_df = df[df['date'] == last_date]
                        col = 'name' if 'name' in df.columns else 'type'
                        
                        def get_v(k):
                            r = today_df[today_df[col].str.contains(k, case=False, na=False)]
                            return (r['buy'].sum() - r['sell'].sum()) / 1000 if not r.empty else 0.0

                        f, t, d = get_v('Foreign'), get_v('Trust'), get_v('Dealer')
                        total = f + t + d
                        score = max(0.97, min(1.03, 1 + (total / 2000) * 0.012))
                        return (float(score), float(total), float(f), float(t), float(d), str(last_date))
        except:
            continue

    # --- 備援機制：yfinance 量能慣性估算 ---
    try:
        ytick = yf.Ticker(f"{pure_id}.TWO" if int(pure_id) > 1000 else f"{pure_id}.TW")
        h = ytick.history(period="5d")
        if not h.empty:
            vol_status = h['Volume'].iloc[-1] / h['Volume'].mean()
            return (1.01 if vol_status > 1 else 1.0, 0.0, 0.0, 0.0, 0.0, f"{h.index[-1].strftime('%m/%d')} (量能估算)")
    except:
        pass
    return default_res

# --- [Session State 狀態初始化] ---
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

# =========================================================
# 主程式路由分流
# =========================================================

# --- 【HOME：首頁導覽】 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統 Pro")
    st.markdown("歡迎使用整合式全方位交易輔助系統，請點選下方核心功能進入專屬決策儀表板：")
    st.divider()
    
    col_a, col_b, col_c, col_d = st.columns(4)
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
    with col_d:
        if st.button("🆘 拯救套牢", use_container_width=True):
            st.session_state.mode = "rescue"
            st.rerun()

# --- 【REALTIME：盤中即時量價頁面】 ---
elif st.session_state.mode == "realtime":
    st.title("⚡ 盤中即時量價（當沖監控）")
    if st.button("⬅️ 返回首頁"): 
        st.session_state.mode = "home"
        st.rerun()
    st.divider()
    
    now = datetime.now(tw_tz)
    # 交易時間判斷：週一至週五 09:00 ~ 13:30
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
                st.info("💡 盤中 AI 建議點位將於台股開盤時間 (09:00 - 13:30) 自動啟動即時動態校正。")
            else:
                st.success(f"🟢 【盤中 AI 動態監控中】數據隨量價即時校正")

            recent_std = df['Close'].tail(15).std()
            avg_vol = df['Volume'].tail(10).mean()
            instant_vol_factor = df['Volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1.0
            clean_name = name.split('(')[0].split('-')[0].strip()

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
            
            stability_index = df['Close'].tail(5).std() / recent_std
            confidence_shield = max(1.0, min(2.0, stability_index))
            vol_expansion = np.sqrt(instant_vol_factor) 
            
            dynamic_offset_low = recent_std * (confidence_shield / vol_expansion)
            dynamic_offset_high = recent_std * (vol_expansion * confidence_shield)
            
            buy_support = curr_price - dynamic_offset_low
            sell_resist = curr_price + dynamic_offset_high

            tick = get_tick_size(curr_price)
            buy_point = round(buy_support / tick) * tick
            sell_target = round(sell_resist / tick) * tick
            expected_return = (sell_target - buy_point) / buy_point * 100

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
                        <b style="color:#22543D; font-size:14px;">📈 預期報酬範圍</b>
                        <h2 style="color:#2F855A; margin:10px 0;">{expected_return:.2f}%</h2>
                    </div>
                """, unsafe_allow_html=True)
                
            if expected_return < 1.2:
                st.info("💡 目前即時市場波動度極低，建議靜待成交量能噴發放大後再進場布局。")

# --- 【FORECAST：波段預估頁面 (布林通道 × 波浪理論)】 ---
elif st.session_state.mode == "forecast":
    if st.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
        
    st.title("📊 波段預估 (布林通道 × 波浪理論精準對齊版)")
    stock_id = st.text_input("輸入代碼 (例: 2330)")

    if stock_id:
        with st.spinner('AI 雙軌指標計算、波浪轉折與 100 日軌道回測中...'):
            df, sym = fetch_stock_data(stock_id, period="150d")
            
            if not df.empty:
                df = df.ffill()
                name = get_stock_name(stock_id)
                
                if len(df) < 100:
                    st.error("❌ 該標的歷史數據不足 100 天，無法執行精密波段與通道回測。")
                else:
                    curr_c = float(df['Close'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    if curr_c == prev_close and len(df) > 2:
                        prev_close = float(df['Close'].iloc[-3])
                    
                    price_diff = curr_c - prev_close
                    active_color = "#E53E3E" if price_diff >= 0 else "#38A169"
                    clean_name = name.split('(')[0].split('-')[0].strip()

                    # --- [布林通道核心計算] ---
                    df['BB_MA'] = df['Close'].rolling(window=20).mean()
                    df['BB_STD'] = df['Close'].rolling(window=20).std()
                    df['BB_Upper'] = df['BB_MA'] + (2 * df['BB_STD'])
                    df['BB_Lower'] = df['BB_MA'] - (2 * df['BB_STD'])
                    
                    latest_bb = df.iloc[-1]
                    
                    if curr_c >= latest_bb['BB_Upper']:
                        bb_pred_target = latest_bb['BB_MA']
                        bb_status = "股價觸及上軌（處於超買區，預期將向中軌均線回檔修正）"
                    elif curr_c <= latest_bb['BB_Lower']:
                        bb_pred_target = latest_bb['BB_MA']
                        bb_status = "股價觸及下軌（處於超賣區，預期將向中軌均線反彈回升）"
                    else:
                        bb_slope = latest_bb['BB_MA'] - df['BB_MA'].iloc[-2]
                        bb_pred_target = curr_c + (bb_slope * 5)
                        bb_status = "通道內常態震盪（跟隨中軌中長期趨勢發展）"

                    # --- [波浪理論結構分析] ---
                    backtest_df = df.tail(100)
                    p_min_idx = backtest_df['Close'].idxmin()
                    p_max_idx = backtest_df['Close'].idxmax()
                    
                    wave_low = backtest_df.loc[p_min_idx, 'Close']
                    wave_high = backtest_df.loc[p_max_idx, 'Close']
                    wave_height = wave_high - wave_low

                    if p_min_idx < p_max_idx:
                        wave_pred_low = wave_high + (wave_height * 0.382)
                        wave_pred_high = wave_high + (wave_height * 0.618)
                        wave_status = "結構歸類：【上升主升浪】。正處於多頭推升擴展結構。"
                    else:
                        wave_pred_low = wave_high - (wave_height * 0.618)
                        wave_pred_high = wave_high - (wave_height * 0.382)
                        wave_status = "結構歸類：【修正調整浪】。正處於 ABC 波段中期修正調整。"

                    # 對齊台股升降單位
                    tick = get_tick_size(curr_c)
                    final_target_min = round(min(bb_pred_target, wave_pred_low) / tick) * tick
                    final_target_max = round(max(bb_pred_target, wave_pred_high) / tick) * tick
                    if final_target_min == final_target_max:
                        final_target_max += tick

                    # --- [歷史 100 日命中率真實回測] ---
                    hit_counts = 0
                    for i in range(len(df) - 100, len(df)):
                        hist_close = df['Close'].iloc[i]
                        hist_upper = df['BB_Upper'].iloc[i]
                        hist_lower = df['BB_Lower'].iloc[i]
                        if hist_lower <= hist_close <= hist_upper:
                            hit_counts += 1
                    bb_accuracy = (hit_counts / 100) * 100

                    # 畫面排版
                    st.markdown(f"""
                        <div style='background: #FFFFFF; padding: 20px; border-radius: 15px; border-left: 10px solid {active_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                            <h2 style='color: #1E293B; margin: 0; font-size: 24px;'>{clean_name} ({stock_id}) 最新行情</h2>
                            <div style='display: flex; align-items: baseline; flex-wrap: wrap;'>
                                <b class='main-price' style='font-size: 65px; color: {active_color}; letter-spacing: -2px;'>{curr_c:.2f}</b>
                                <div style='margin-left: 15px;'>
                                    <span style='font-size: 24px; color: {active_color}; font-weight: 900; display: block;'>
                                        {'▲' if price_diff >= 0 else '▼'} {abs(price_diff):.2f}
                                    </span>
                                    <span style='font-size: 16px; color: {active_color}; font-weight: 700;'>
                                        ({(price_diff/prev_close*100):.2f}%)
                                    </span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1E1B4B 0%, #311042 100%); padding: 25px; border-radius: 16px; color: white; margin-top: 20px; box-shadow: 0 6px 15px rgba(0,0,0,0.15);">
                            <span style="font-size: 14px; color: #C7D2FE; font-weight: bold; letter-spacing: 1px;">🔮 AI 雙軌預估未來波段目標價區間</span>
                            <div style="font-size: 46px; font-weight: 800; margin: 10px 0; color: #FBBF24;">
                                {final_target_min:.2f} <span style="font-size: 26px; color: #A5B4FC; font-weight: 300;">至</span> {final_target_max:.2f}
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #E0E7FF; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 12px;">
                                <span>🎯 布林狀態：{bb_status}</span>
                                <span>📈 歷史 100 日軌道覆蓋勝率：<b>{bb_accuracy:.1f}%</b></span>
                            </div>
                            <div style="font-size: 13px; color: #E0E7FF; margin-top: 5px;">
                                🌊 波浪診斷：{wave_status} (100日極值參考點：低點 {wave_low:.2f} / 高點 {wave_high:.2f})
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.divider()

                    # --- [視覺化圖表繪製] ---
                    st.subheader("📈 布林通道軌道與波浪極值轉折追蹤")
                    plot_df = df.tail(100)
                    
                    fig, ax = plt.subplots(figsize=(11, 5.5))
                    ax.plot(plot_df.index, plot_df['Close'], label='Close Price (收盤價)', color='#1E293B', linewidth=2, zorder=3)
                    ax.plot(plot_df.index, plot_df['BB_MA'], label='BB Middle (中軌線)', color='#3B82F6', linestyle='--', alpha=0.8)
                    ax.plot(plot_df.index, plot_df['BB_Upper'], label='BB Upper (上軌壓力)', color='#EF4444', alpha=0.6, linewidth=1.2)
                    ax.plot(plot_df.index, plot_df['BB_Lower'], label='BB Lower (下軌支撐)', color='#10B981', alpha=0.6, linewidth=1.2)
                    
                    ax.fill_between(plot_df.index, plot_df['BB_Lower'], plot_df['BB_Upper'], color='#3B82F6', alpha=0.04)

                    # 標註高低極值點與向量
                    ax.scatter(p_min_idx, wave_low, color='#10B981', s=120, marker='^', label='100D Wave Low (波浪起點)', zorder=5)
                    ax.scatter(p_max_idx, wave_high, color='#EF4444', s=120, marker='v', label='100D Wave High (波浪頂點)', zorder=5)
                    ax.plot([p_min_idx, p_max_idx], [wave_low, wave_high], color='#F59E0B', linestyle=':', linewidth=1.8, label='Wave Vector')

                    ax.set_title(f"{stock_id} Bollinger Bands & Elliott Wave Dashboard", fontsize=11, fontweight='bold', pad=12)
                    ax.grid(True, linestyle=':', alpha=0.5)
                    ax.legend(loc='upper left', frameon=True, fontsize=9)
                    plt.xticks(rotation=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.caption("💡 註：本圖表橫軸為歷史交易日期，灰色帶狀區間為標準布林震盪軌道，金黃虛線為多空波浪趨勢向量。")

# --- 【SECTOR：類群輪動預警頁面】 ---
elif st.session_state.mode == "sector":
    st.title("💎 類群輪動預警儀表板")
    if st.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
    st.divider()
    
    st.markdown("### 目前系統即時監控：PCB、記憶體、AI 供應鏈、重電綠能與傳統產業群組")
    
    with st.spinner('AI 正在計算各細分產業鏈大戶資金流入強度...'):
        flow_report = []
        for en_id, tickers in INDUSTRY_CHAINS_EN.items():
            try:
                data = yf.download(tickers, period="10d", progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1).mean() * 100
                    vol_ratio = data['Volume'].iloc[-1].sum() / data['Volume'].tail(5).mean().sum()
                    flow_report.append({"ID": en_id, "漲跌%": ret, "資金流入": vol_ratio})
            except: 
                continue
        
        df_flow = pd.DataFrame(flow_report)

        if not df_flow.empty:
            # 強勢主流焦點判斷
            buy_candidates = df_flow[(df_flow['資金流入'] > 1.2) & (df_flow['漲跌%'] > -0.5)]
            
            st.subheader("🎯 當前強勢主流族群")
            if not buy_candidates.empty:
                best_sector_id = buy_candidates.sort_values(by="資金流入", ascending=False).iloc[0]['ID']
                st.success(f"🚀 **【大戶強烈聚焦關注】：{name_map[best_sector_id]}**")
                st.info(f"💡 理由：該產業今日資金流入強度高達 {buy_candidates['資金流入'].max():.2f} 倍，顯示大戶籌碼進場意願極高，具備起漲發動動能。")
                
                strong_tickers = INDUSTRY_CHAINS_EN.get(best_sector_id, [])
                if strong_tickers:
                    st.write(f"🔍 **{name_map[best_sector_id]} 領頭概念股表現：**")
                    s_cols = st.columns(len(strong_tickers))
                    for idx, ticker in enumerate(strong_tickers):
                        try:
                            s_data = yf.Ticker(ticker)
                            s_name = get_stock_name(ticker.split('.')[0]).replace("走勢圖", "").strip()
                            s_price_df = s_data.history(period="2d")
                            if len(s_price_df) >= 2:
                                s_ret = (s_price_df['Close'].iloc[-1] / s_price_df['Close'].iloc[-2] - 1) * 100
                                with s_cols[idx]:
                                    st.metric(label=s_name, value=f"{s_price_df['Close'].iloc[-1]:.2f}", delta=f"{s_ret:.2f}%")
                        except: continue
            else:
                st.warning("⚠️ 目前大盤多數細分產業處於縮量或盤整期，暫無「爆量突破」標的，建議保留現金分批布局。")

            st.divider()
            
            # 低基期補漲潛力產業
            low_base_candidates = df_flow[(df_flow['資金流入'] > 1.05) & (df_flow['資金流入'] < 1.8) & (df_flow['漲跌%'] >= -1.0) & (df_flow['漲跌%'] <= 2.5)]
            st.subheader("🎯 AI 低基期潛力產業建議")
            if not low_base_candidates.empty:
                best_bet = low_base_candidates.sort_values(by="資金流入", ascending=False).iloc[0]
                best_id = best_bet['ID']
                st.success(f"🚀 **【潛力安全補漲關注】：{name_map[best_id]}**")
                
                target_tickers = INDUSTRY_CHAINS_EN.get(best_id, [])
                if target_tickers:
                    st.write(f"💡 **產業鏈精選標的：**")
                    cols = st.columns(len(target_tickers))
                    for idx, ticker in enumerate(target_tickers):
                        try:
                            t_data = yf.Ticker(ticker)
                            raw_name = get_stock_name(ticker.split('.')[0])
                            t_name = raw_name.replace("走勢圖", "").replace("Yahoo奇摩股市", "").strip()
                            t_price = t_data.history(period="2d")
                            if len(t_price) >= 2:
                                t_ret = (t_price['Close'].iloc[-1] / t_price['Close'].iloc[-2] - 1) * 100
                                with cols[idx]:
                                    st.metric(label=t_name, value=f"{t_price['Close'].iloc[-1]:.2f}", delta=f"{t_ret:.2f}%")
                        except: continue
            
            st.divider()
            
            # 詳細表格
            st.write("📋 **產業資金流向數據明細**")
            df_display = df_flow.copy()
            df_display['產業名稱'] = df_display['ID'].map(name_map)
            st.dataframe(df_display[['產業名稱', '漲跌%', '資金流入']].sort_values(by='資金流入', ascending=False), use_container_width=True, hide_index=True)

            # 資金流入排行榜圖表
            st.write("📈 **Sector Money Flow (資金流入強度排行榜)**")
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df_flow.sort_values(by="資金流入")
            ax.barh(df_plot['ID'], df_plot['資金流入'], color='gold', edgecolor='black', height=0.6)
            ax.axvline(x=1.0, color='red', ls='--', alpha=0.6, label='Baseline (均量線)')
            ax.set_title("Industry Sector Money Flow Intensity", fontsize=10, fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            
            # 中文對照註解
            st.markdown("#### 📘 分類代碼英漢對照註解 (Legends):")
            c1, c2 = st.columns(2)
            sorted_en_ids = df_plot['ID'].tolist()[::-1]
            for i, en_id in enumerate(sorted_en_ids):
                with (c1 if i % 2 == 0 else c2):
                    st.write(f"- **{en_id}**: {name_map[en_id]}")
        else:
            st.error("暫時無法取得產業板塊數據，請確認 API 連線狀態。")

# --- 【RESCUE：拯救套牢診斷頁面】 ---
elif st.session_state.mode == "rescue":
    st.title("🆘 拯救套牢診斷艙")
    if st.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"
        st.rerun()
    st.divider()
    
    st.markdown("### 🔍 庫存套牢風險診斷與攤平計算")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        r_id = st.text_input("請輸入套牢股票代碼：")
        r_cost = st.number_input("您的買進持股成本價：", min_value=0.0, step=0.1)
    with col_r2:
        r_volume = st.number_input("持有張數 (張)：", min_value=0, step=1)
        
    if r_id and r_cost > 0 and r_volume > 0:
        with st.spinner('正在分析該股基本面壓力與防守支撐點...'):
            df, sym = fetch_stock_data(r_id, period="100d")
            if not df.empty:
                df = df.ffill()
                curr_p = float(df['Close'].iloc[-1])
                loss_pct = ((curr_p - r_cost) / r_cost) * 100
                
                st.subheader("📊 庫存損益現狀診斷")
                if loss_pct >= 0:
                    st.success(f"🎉 帳面上目前為獲利狀態！報酬率：+{loss_pct:.2f}%。建議順勢移動停利。")
                else:
                    st.error(f"❌ 目前處於套牢狀態。報酬率：{loss_pct:.2f}% (現價：{curr_p:.2f})")
                    
                    # 計算支撐與壓力作為技術參考
                    support_p = df['Low'].tail(20).min()
                    resistance_p = df['High'].tail(20).max()
                    
                    st.markdown(f"""
                        > **⚠️ 專家技術面診斷提示：**
                        > - 近20日波段**強勁支撐位**：<b style="color:#28A745;">{support_p:.2f}</b> (若此價位跌破，代表弱勢破底，不建議盲目攤平)
                        > - 近20日波段**解套壓力位**：<b style="color:#DC3545;">{resistance_p:.2f}</b>
                    """, unsafe_allow_html=True)
                    
                    st.divider()
                    st.subheader("🧮 智慧型分批解套解盲 (加碼攤平模擬)")
                    
                    # 模擬加碼
                    add_shares = st.slider("預計加碼買進張數 (張)：", min_value=1, max_value=r_volume * 3, value=r_volume)
                    
                    new_cost = ((r_cost * r_volume) + (curr_p * add_shares)) / (r_volume + add_shares)
                    new_loss_pct = ((curr_p - new_cost) / new_cost) * 100
                    
                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        st.metric(label="攤平後平均成本價", value=f"{new_cost:.2f}", delta=f"成本降低 {r_cost - new_cost:.2f}")
                    with c_m2:
                        st.metric(label="攤平後預估新報酬率", value=f"{new_loss_pct:.2f}%", delta=f"風險縮減 {abs(loss_pct) - abs(new_loss_pct):.2f}%")
                        
                    st.info(f"💡 策略建議：如果決定進行加碼攤平，最佳加碼時間點為股價成功於支撐線 {support_p:.2f} 附近放量止跌、或是布林通道下軌出現長下影線時分批建立。")
            else:
                st.error("❌ 無法取得此股票代碼之市場歷史數據。")
