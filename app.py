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

# --- [全域初始化與網頁設定] ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股 AI 交易助手 Pro", layout="wide", page_icon="💹")

# 設定時區
tw_tz = pytz.timezone("Asia/Taipei")

# --- [字體防亂碼全英設定] ---
# 圖表標籤與圖例全部改用標準英文，100% 避免 Streamlit Cloud 出現豆腐塊
matplotlib.rcParams['axes.unicode_minus'] = False 

# --- [全域變態變數與金鑰] ---
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
def fetch_stock_data(stock_id, period="120d"):
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

def fetch_finmind_chips(stock_id, token=FINMIND_TOKEN):
    default_res = (1.0, 0.0, 0.0, 0.0, 0.0, "備援演算啟動")
    pure_id = str(stock_id).split('.')[0]
    id_variants = [pure_id, f"{pure_id}.TW"]
    
    for target_id in id_variants:
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            start_dt = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            params = {"dataset": "InstitutionalInvestorsBuySell", "data_id": target_id, "start_date": start_dt, "token": token}
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
    try:
        ytick = yf.Ticker(f"{pure_id}.TWO" if int(pure_id) > 1000 else f"{pure_id}.TW")
        h = ytick.history(period="5d")
        if not h.empty:
            vol_status = h['Volume'].iloc[-1] / h['Volume'].mean()
            return (1.01 if vol_status > 1 else 1.0, 0.0, 0.0, 0.0, 0.0, f"{h.index[-1].strftime('%m/%d')} (量能估算)")
    except:
        pass
    return default_res

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

# =========================================================
# 主程式分流
# =========================================================

# --- 【HOME：首頁導覽】 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統 Pro")
    st.markdown("歡迎使用整合式全方位交易輔助系統，請點選下方核心功能進入專屬決策儀表板：")
    st.divider()
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True): st.session_state.mode = "realtime"; st.rerun()
    with col_b:
        if st.button("📊 波段預估", use_container_width=True): st.session_state.mode = "forecast"; st.rerun()
    with col_c:
        if st.button("💎 類群輪動預警", use_container_width=True): st.session_state.mode = "sector"; st.rerun()
    with col_d:
        if st.button("🆘 拯救套牢", use_container_width=True): st.session_state.mode = "rescue"; st.rerun()

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

# --- 【FORECAST：波段預估頁面 (斐波那契波段擴展滿足點版)】 ---
elif st.session_state.mode == "forecast":
    if st.button("⬅️ 返回首頁"): st.session_state.mode = "home"; st.rerun()
        
    st.title("📊 波段未來趨勢目標價預估 (黃金分割擴展版)")
    stock_id = st.text_input("輸入代碼 (例: 2330)")

    if stock_id:
        with st.spinner('AI 技術面趨勢量能與未來目標滿足點計算中...'):
            df, sym = fetch_stock_data(stock_id, period="150d")
            
            if not df.empty:
                df = df.ffill()
                name = get_stock_name(stock_id)
                
                if len(df) < 100:
                    st.error("❌ 該標的歷史數據不足 100 天，無法執行精密波段趨勢演算。")
                else:
                    curr_c = float(df['Close'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    price_diff = curr_c - prev_close
                    active_color = "#E53E3E" if price_diff >= 0 else "#38A169"
                    clean_name = name.split('(')[0].split('-')[0].strip()

                    # --- [布林通道計算] ---
                    df['BB_MA'] = df['Close'].rolling(window=20).mean()
                    df['BB_STD'] = df['Close'].rolling(window=20).std()
                    df['BB_Upper'] = df['BB_MA'] + (2 * df['BB_STD'])
                    df['BB_Lower'] = df['BB_MA'] - (2 * df['BB_STD'])

                    # --- [黃金分割擴展波浪模型演算法] ---
                    backtest_df = df.tail(100)
                    p_min_idx = backtest_df['Close'].idxmin()
                    p_max_idx = backtest_df['Close'].idxmax()
                    
                    wave_low = backtest_df.loc[p_min_idx, 'Close']
                    wave_high = backtest_df.loc[p_max_idx, 'Close']
                    wave_height = wave_high - wave_low

                    # 判斷多空大結構，決定擴展方向
                    tick = get_tick_size(curr_c)
                    if p_min_idx < p_max_idx:
                        # 【多頭推升浪】目標價：以底部向上過高後的擴展 1.382 ~ 1.618 倍計算
                        target_low = wave_low + (wave_height * 1.382)
                        target_high = wave_low + (wave_height * 1.618)
                        trend_status = "📈 多頭結構：目前正處於「主升段推升結構」，若突破前高，AI 預估未來強勢波段目標滿足點。"
                    else:
                        # 【空頭修正/反彈浪】目標價：以高點向下的反彈 0.382 ~ 0.618 阻力區
                        target_low = wave_low + (wave_height * 0.382)
                        target_high = wave_low + (wave_height * 0.618)
                        trend_status = "📉 修正結構：目前處於「波段ABC修正/築底階段」，AI 預估未來中線強彈之壓力波段區間。"

                    # 對齊台股升降單位
                    final_target_min = round(target_low / tick) * tick
                    final_target_max = round(target_high / tick) * tick
                    if final_target_min == final_target_max:
                        final_target_max += tick

                    # --- [真實命中軌道率回測] ---
                    hit_counts = sum(1 for i in range(len(df) - 100, len(df)) if df['BB_Lower'].iloc[i] <= df['Close'].iloc[i] <= df['BB_Upper'].iloc[i])
                    bb_accuracy = (hit_counts / 100) * 100

                    # 畫面排版
                    st.markdown(f"""
                        <div style='background: #FFFFFF; padding: 20px; border-radius: 15px; border-left: 10px solid {active_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                            <h2 style='color: #1E293B; margin: 0; font-size: 24px;'>{clean_name} ({stock_id}) 最新行情 (現價)</h2>
                            <div style='display: flex; align-items: baseline; flex-wrap: wrap;'>
                                <b style='font-size: 65px; color: {active_color}; letter-spacing: -2px;'>{curr_c:.2f}</b>
                                <div style='margin-left: 15px;'>
                                    <span style='font-size: 24px; color: {active_color}; font-weight: 900; display: block;'>{'▲' if price_diff >= 0 else '▼'} {abs(price_diff):.2f}</span>
                                    <span style='font-size: 16px; color: {active_color}; font-weight: 700;'>({(price_diff/prev_close*100):.2f}%)</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); padding: 25px; border-radius: 16px; color: white; margin-top: 20px; box-shadow: 0 6px 15px rgba(0,0,0,0.15);">
                            <span style="font-size: 14px; color: #93C5FD; font-weight: bold; letter-spacing: 1px;">🔮 AI 預估未來中長期波段到達目標價區間</span>
                            <div style="font-size: 50px; font-weight: 800; margin: 10px 0; color: #F59E0B;">
                                {final_target_min:.2f} <span style="font-size: 26px; color: #93C5FD; font-weight: 300;">至</span> {final_target_max:.2f}
                            </div>
                            <div style="font-size: 14px; color: #E2E8F0; border-top: 1px solid rgba(255,255,255,0.15); padding-top: 12px;">
                                🎯 趨勢狀態：{trend_status}
                            </div>
                            <div style="font-size: 13px; color: #94A3B8; margin-top: 5px;">
                                📐 演算依據：100日波段大底部參考點：{wave_low:.2f} ｜ 歷史波動軌道覆蓋勝率：{bb_accuracy:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.divider()

                    # --- [全英文圖表繪製] ---
                    st.subheader("📈 技術指標與趨勢波浪軌道追蹤 (全英圖例)")
                    plot_df = df.tail(100)
                    fig, ax = plt.subplots(figsize=(11, 5.5))
                    ax.plot(plot_df.index, plot_df['Close'], label='Close Price', color='#1E293B', linewidth=2, zorder=3)
                    ax.plot(plot_df.index, plot_df['BB_MA'], label='BB Middle (20 MA)', color='#3B82F6', linestyle='--', alpha=0.8)
                    ax.plot(plot_df.index, plot_df['BB_Upper'], label='BB Upper (2 Std)', color='#EF4444', alpha=0.6, linewidth=1.2)
                    ax.plot(plot_df.index, plot_df['BB_Lower'], label='BB Lower (2 Std)', color='#10B981', alpha=0.6, linewidth=1.2)
                    ax.fill_between(plot_df.index, plot_df['BB_Lower'], plot_df['BB_Upper'], color='#3B82F6', alpha=0.04)

                    ax.scatter(p_min_idx, wave_low, color='#10B981', s=120, marker='^', label='100D Wave Low Base', zorder=5)
                    ax.scatter(p_max_idx, wave_high, color='#EF4444', s=120, marker='v', label='100D Wave High Peak', zorder=5)
                    ax.plot([p_min_idx, p_max_idx], [wave_low, wave_high], color='#F59E0B', linestyle=':', linewidth=1.8, label='Wave Trend Vector')

                    ax.set_title(f"{stock_id} Bollinger Bands & Fibonacci Extension Forecast", fontsize=12, fontweight='bold', pad=12)
                    ax.grid(True, linestyle=':', alpha=0.5)
                    ax.legend(loc='upper left', frameon=True, fontsize=9)
                    plt.xticks(rotation=15)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- [前端中文對照註解說明區] ---
                    st.markdown("---")
                    st.markdown("### 📘 儀表板技術指標與全英圖例對照表")
                    info_data = {
                        "圖表項目 (Legend)": ["Close Price", "BB Middle (20 MA)", "BB Upper / Lower", "100D Wave Low Base", "100D Wave High Peak", "Wave Trend Vector"],
                        "中文註解說明": ["每日股票收盤價", "布林通道中軌 (20日移動平均線，生命線)", "布林通道上軌壓力線 / 下軌支撐線", "過去100天歷史大底部（黃金分割計算基點）", "過去100天歷史大頂部（黃金分割計算高點）", "波段高低點之多空主趨勢引導線"]
                    }
                    st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

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
                            s_data = yf.Ticker(ticker); s_name = get_stock_name(ticker.split('.')[0]).replace("走勢圖", "").strip()
                            s_price_df = s_data.history(period="2d")
                            if len(s_price_df) >= 2:
                                s_ret = (s_price_df['Close'].iloc[-1] / s_price_df['Close'].iloc[-2] - 1) * 100
                                with s_cols[idx]: st.metric(label=s_name, value=f"{s_price_df['Close'].iloc[-1]:.2f}", delta=f"{s_ret:.2f}%")
                        except: continue
            else:
                st.warning("⚠️ 目前大盤多數細分產業處於縮量或盤整期，暫無爆量突破標的。")

            st.divider()
            df_display = df_flow.copy()
            df_display['產業名稱'] = df_display['ID'].map(name_map)
            st.dataframe(df_display[['產業名稱', '漲跌%', '資金流入']].sort_values(by='資金流入', ascending=False), use_container_width=True, hide_index=True)

            st.write("📈 **Sector Money Flow Intensity (全英對照榜)**")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_plot = df_flow.sort_values(by="資金流入")
            ax.barh(df_plot['ID'], df_plot['資金流入'], color='gold', edgecolor='black', height=0.6)
            ax.axvline(x=1.0, color='red', ls='--', alpha=0.6)
            st.pyplot(fig)
            
            st.markdown("#### 📘 分類代碼英漢對照對應表:")
            c1, c2 = st.columns(2)
            for i, en_id in enumerate(df_plot['ID'].tolist()[::-1]):
                with (c1 if i % 2 == 0 else c2): st.write(f"- **{en_id}**: {name_map[en_id]}")

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
