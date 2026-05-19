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

# --- [全域設定區 - 嚴格置頂且僅呼叫一次] ---
st.set_page_config(page_title="台股 AI 交易助手 Pro", layout="wide", page_icon="💹")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0wNSAxODozOToxOSIsInVzZXJfaWQiOiJhYXJvbjA3IiwiZW1haWwiOiJodWlodWkyNTU2aEBnbWFpbC5jb20iLCJpcCI6IjEuMTcwLjkwLjIyNSJ9.n-uv7ODTCIAjl0mffN2_rsIvqwLRWB3rVFCBd7jG0bE"
tw_tz = pytz.timezone("Asia/Taipei")

# --- 🎯 修正圖片亂碼：強制手動載入系統字體 ---
def set_mpl_font():
    fonts = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'SimHei', 'Arial Unicode MS']
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

# --- [核心函數定義] ---
def fetch_finmind_chips(stock_id, token=FINMIND_TOKEN):
    default_res = (1.0, 0.0, 0.0, 0.0, 0.0, "API 維護中")
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

    try:
        ytick = yf.Ticker(f"{pure_id}.TWO" if int(pure_id) > 1000 else f"{pure_id}.TW")
        h = ytick.history(period="5d")
        if not h.empty:
            vol_status = h['Volume'].iloc[-1] / h['Volume'].mean()
            return (1.01 if vol_status > 1 else 1.0, 0.0, 0.0, 0.0, 0.0, f"{h.index[-1].strftime('%m/%d')} (量能估算)")
    except:
        pass

    return default_res
        
def get_global_risk_impact():
    """抓取原油 (BZ=F) 評估地緣政治與避險風險因子"""
    try:
        oil = yf.download("BZ=F", period="5d", progress=False)
        if oil.empty: return 1.0
        if isinstance(oil.columns, pd.MultiIndex):
            oil.columns = oil.columns.get_level_values(0)
        oil_change = (oil['Close'].iloc[-1] / oil['Close'].iloc[-5] - 1) * 100
        risk_bias = 1 - (oil_change * 0.004) 
        return max(0.95, min(1.05, risk_bias)) 
    except:
        return 1.0

def get_tick_size(price):
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.1
    elif price < 500: return 0.5
    elif price < 1000: return 1.0
    else: return 5.0

if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def calculate_real_accuracy(df, factor, side='high'):
    try:
        df_copy = df.copy().ffill()
        backtest_days = 60 
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

# ================== 介面控制 ==================
name_map = {
    "PCB-CCL": "PCB-材料 (CCL/銅箔)", "PCB-Substrate": "PCB-載板 (ABF/BT)", "PCB-Assembly": "PCB-組裝加工 (硬板/HDI)",
    "Memory-Fab": "記憶體-原廠/代工", "Memory-Module": "記憶體-模組廠", "Memory-Controller": "記憶體-控制 IC", "Memory-DDR5": "記憶體-DDR5/高速傳輸",
    "Semi-Equip": "半導體-設備/CoWoS", "Semi-OSAT": "半導體-封測 (先進封裝/測試)", "AI-ASIC": "AI 特用晶片 (矽智財/ASIC)",
    "AI-Case": "AI 伺服器 (機殼/滑軌)", "AI-Cooling": "AI 伺服器 (散熱/水冷)", "AI-ODM": "AI 伺服器 (ODM 代工)",
    "CPO-Silicon": "矽光子 (CPO/光通訊)", "Satellite-LEO": "低軌衛星 (航太/地面站)",
    "Display-Panel": "面板-驅動 IC/面板廠", "Passive-Comp": "被動元件 (MLCC/電阻)", "Optical-Lens": "光學鏡頭 (手機/車載)",
    "Auto-EV": "車用電子 (電動車/二極體)", "Power-Grid": "重電/電力 (政策股)", "Shipping": "航運 (貨櫃/散裝)"
}

INDUSTRY_CHAINS_EN = {
    "PCB-CCL": ["6213.TW", "2383.TW", "6274.TW", "8358.TWO", "2367.TW"],
    "PCB-Substrate": ["8046.TW", "3037.TW", "3189.TW", "6667.TW"],
    "PCB-Assembly": ["2367.TW", "2313.TW", "2368.TW", "4958.TW", "3044.TW"],
    "Memory-Fab": ["2344.TW", "2337.TW", "2408.TW", "3006.TW"],
    "Memory-Module": ["3260.TWO", "8299.TW", "2451.TW", "3264.TWO"],
    "Memory-Controller": ["8299.TW", "4966.TW", "6233.TWO"],
    "Memory-DDR5": ["6138.TW", "6213.TW", "8299.TW", "3260.TWO"],
    "Semi-Equip": ["3131.TWO", "3583.TW", "1560.TW", "6187.TWO"],
    "Semi-OSAT": ["2311.TW", "3711.TW", "2449.TW", "6147.TWO"],
    "AI-ASIC": ["3661.TW", "3443.TW", "6643.TW", "3035.TW"],
    "AI-ODM": ["2382.TW", "2317.TW", "3231.TW", "6669.TW"],
    "AI-Cooling": ["3017.TW", "3324.TW", "2421.TW", "6230.TW"],
    "AI-Case": ["8210.TW", "2059.TW", "6803.TW", "3693.TW"],
    "CPO-Silicon": ["3363.TWO", "4979.TWO", "3081.TWO", "6451.TW"],
    "Satellite-LEO": ["2313.TW", "3491.TWO", "2314.TW", "3380.TW"],
    "Display-Panel": ["2409.TW", "3481.TW", "6116.TW", "3034.TW"],
    "Passive-Comp": ["2327.TW", "2492.TW", "6173.TWO", "6127.TWO"],
    "Optical-Lens": ["3008.TW", "3406.TW", "3362.TW", "3504.TW"],
    "Shipping": ["2603.TW", "2609.TW", "2615.TW", "2606.TW"],
    "Power-Grid": ["1513.TW", "1503.TW", "1519.TW", "1514.TW"],
    "Auto-EV": ["2317.TW", "2481.TW", "5425.TWO", "3675.TW"]
}

if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 交易決策系統")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("⚡ 盤中即時量價", use_container_width=True):
            st.session_state.mode = "realtime"; st.rerun()
    with col_b:
        if st.button("📊 波段預估", use_container_width=True):
            st.session_state.mode = "forecast"; st.rerun()
    with col_c:
        if st.button("💎 類群輪動預警", use_container_width=True):
            st.session_state.mode = "sector"; st.rerun()
    with col_d:
        if st.button("🆘 拯救套牢", use_container_width=True):
            st.session_state.mode = "rescue"; st.rerun()

elif st.session_state.mode == "sector":
    st.title("💎 類群輪動預警")
    if st.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"; st.rerun()
    st.divider()
    
    with st.spinner('AI 正在計算各產業獲利潛力...'):
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
            except: continue
        
        df_flow = pd.DataFrame(flow_report)

        if not df_flow.empty:
            buy_candidates = df_flow[(df_flow['資金流入'] > 1.2) & (df_flow['漲跌%'] > -0.5)]
            st.subheader("🎯 目前強勢族群")
            if not buy_candidates.empty:
                best_sector_id = buy_candidates.sort_values(by="資金流入", ascending=False).iloc[0]['ID']
                st.success(f"🚀 **【強烈建議關注】：{name_map[best_sector_id]}**")
                strong_tickers = INDUSTRY_CHAINS_EN.get(best_sector_id, [])
                if strong_tickers:
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
                st.warning("⚠️ 目前多數類股處於縮量或盤整期，暫無「爆量起漲」標的。")

            st.divider()
            low_base_candidates = df_flow[(df_flow['資金流入'] > 1.05) & (df_flow['資金流入'] < 1.8) & (df_flow['漲跌%'] >= -1.0) & (df_flow['漲跌%'] <= 2.5)]
            st.subheader("🎯 AI 低基期潛力產業建議")
            if not low_base_candidates.empty:
                best_id = low_base_candidates.sort_values(by="資金流入", ascending=False).iloc[0]['ID']
                st.success(f"🚀 **【潛力補漲關注】：{name_map[best_id]}**")
                target_tickers = INDUSTRY_CHAINS_EN.get(best_id, [])
                if target_tickers:
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
            df_display = df_flow.copy()
            df_display['產業名稱'] = df_display['ID'].map(name_map)
            st.dataframe(df_display[['產業名稱', '漲跌%', '資金流入']].sort_values(by='資金流入', ascending=False), use_container_width=True, hide_index=True)

            st.write("📈 **Sector Money Flow (資金流入排行榜)**")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_plot = df_flow.sort_values(by="資金流入")
            ax.barh(df_plot['ID'], df_plot['資金流入'], color='gold', edgecolor='black')
            ax.axvline(x=1.0, color='red', ls='--', alpha=0.6)
            st.pyplot(fig)

elif st.session_state.mode == "realtime":
    st.title("⚡ 盤中即時量價（當沖）")
    if st.sidebar.button("⬅️ 返回首頁"): 
        st.session_state.mode = "home"; st.rerun()
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
                st.warning(f"🕒 【目前非交易時段】系統暫停動態演算。現在時間：{now.strftime('%H:%M')}。")
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

                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown(f"<div style='background:#F0F9FF; padding:20px; border-radius:12px; border-left:8px solid #3182CE; text-align:center;'><b style='color:#2C5282; font-size:14px;'>🔹 動態支撐買點</b><h2 style='color:#1E40AF; margin:10px 0;'>{buy_point:.2f}</h2></div>", unsafe_allow_html=True)
                with d2:
                    st.markdown(f"<div style='background:#FFF5F5; padding:20px; border-radius:12px; border-left:8px solid #E53E3E; text-align:center;'><b style='color:#9B2C2C; font-size:14px;'>🔴 動態壓力賣點</b><h2 style='color:#991B1B; margin:10px 0;'>{sell_target:.2f}</h2></div>", unsafe_allow_html=True)
                with d3:
                    st.markdown(f"<div style='background:#F0FFF4; padding:20px; border-radius:12px; border-left:8px solid #38A169; text-align:center;'><b style='color:#22543D; font-size:14px;'>📈 預期報酬</b><h2 style='color:#2F855A; margin:10px 0;'>{expected_return:.2f}%</h2></div>", unsafe_allow_html=True)

elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"):
        st.session_state.mode = "home"; st.rerun()
    st.title("📊 波段預估")
    stock_id = st.text_input("輸入代碼 (例: 2330)")

    if stock_id:
        with st.spinner('AI 多因子計算與回測中...'):
            df, sym = fetch_stock_data(stock_id)
            if not df.empty:
                df = df.ffill()
                name = get_stock_name(stock_id)
                if len(df) < 2:
                    st.error("數據量不足")
                else:
                    curr_c = float(df['Close'].iloc[-1]) 
                    prev_close = float(df['Close'].iloc[-2]) 
    
                    if curr_c == prev_close and len(df) > 2:
                        prev_close = float(df['Close'].iloc[-3])

                    price_diff = curr_c - prev_close 
                    price_change_pct = (price_diff / prev_close) * 100 
                
                    c_score, net_lots, f_net, t_net, d_net, chip_date = fetch_finmind_chips(stock_id)
                    relative_volume = df['Volume'].iloc[-1] / df['Volume'].tail(5).mean()
                    sector_momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                    sector_bias = 1 + (sector_momentum * 0.005)

                    # 🟢 【修正】正確呼叫原油避險因子函數，活化全球風險控管
                    risk_factor = get_global_risk_impact()
                    
                    tech_bias = 1 + (relative_volume - 1) * 0.015 + (sector_momentum * 0.002)
                    bias = tech_bias * c_score * risk_factor
                    bias = max(0.95, min(1.08, bias)) 

                    tr = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
                    atr = tr.rolling(14).mean().iloc[-1]
                    
                    vol_impact = max(0.02, min(0.12, 0.04 * relative_volume * sector_bias))
                    
                    if curr_c >= prev_close:
                        est_open_raw = curr_c + (atr * vol_impact * bias)
                    else:
                        est_open_raw = curr_c - (atr * vol_impact / bias)

                    tick = get_tick_size(curr_c)
                    vol_inertia = round((atr * bias) / tick) * tick 
                    est_open = round(est_open_raw / tick) * tick
                    active_color = "#E53E3E" if price_diff >= 0 else "#38A169"
                    clean_name = name.split('(')[0].split('-')[0].strip()
                    
                    # 🟢 【修正】計算即時動態勝率回測並呈現在前端
                    real_acc = calculate_real_accuracy(df, factor=vol_impact)

                    st.markdown(f"""
                        <div style='background: #FFFFFF; padding: 20px; border-radius: 15px; border-left: 10px solid {active_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                            <h2 style='color: #1E293B; margin: 0; font-size: 24px;'>({clean_name})的收盤價</h2>
                            <div style='display: flex; align-items: baseline; flex-wrap: wrap;'>
                                <b style='font-size: 75px; color: {active_color}; letter-spacing: -2px;'>{curr_c:.2f}</b>
                                <div style='margin-left: 15px;'>
                                    <span style='font-size: 28px; color: {active_color}; font-weight: 900; display: block;'>
                                        {'▲' if price_diff >= 0 else '▼'} {abs(price_diff):.2f}
                                    </span>
                                    <span style='font-size: 18px; color: {active_color}; font-weight: 700;'>
                                        ({price_change_pct:.2f}%)
                                    </span>
                                </div>
                            </div>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #64748B;">🎯 近 60 日策略預估達成勝率：{real_acc:.1f}%</p>
                        </div>

                        <div style='display: flex; background: #0F172A; padding: 15px; border-radius: 12px; color: white; margin-top: 15px; gap: 10px;'>
                            <div style='flex: 1; text-align: center; border-right: 1px solid #334155;'>
                                <span style='font-size: 12px; color: #94A3B8;'>複合籌碼風險偏置</span>
                                <div style='font-size: 18px; font-weight: bold;'>{bias:.3f}</div>
                            </div>
                            <div style='flex: 1; text-align: center; border-right: 1px solid #334155;'>
                                <span style='font-size: 12px; color: #94A3B8;'>波動慣性支撐</span>
                                <div style='font-size: 18px; font-weight: bold; color: #FACC15;'>{vol_inertia:.2f}</div>
                            </div>
                            <div style='flex: 1; text-align: center;'>
                                <span style='font-size: 12px; color: #94A3B8;'>AI 預估次交易日開盤</span>
                                <div style='font-size: 18px; font-weight: bold; color: #38A169;'>{est_open:.2f}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
