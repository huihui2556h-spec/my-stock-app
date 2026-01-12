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
# 1. 系統配置與視覺初始化
# =========================================================
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

# 初始化頁面模式
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 多因子運算引擎 (FinMind 籌碼 + 國際盤 + 波動慣性)
# =========================================================

# --- 🎯 籌碼面：FinMind 法人籌碼權重 ---
def get_chip_factor(stock_id):
    """計算法人籌碼修正因子：考量近五日法人買賣超淨額"""
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取近 15 天以確保扣除假日有 5 個交易日
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            # 合計三大法人近五日買賣超
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net_buy > 0:
                return 1.025, "✅ 籌碼面：法人偏多 (近五日買超)"
            else:
                return 0.975, "⚠️ 籌碼面：法人偏空 (近五日賣超)"
    except:
        pass
    return 1.0, "ℹ️ 籌碼面：中性 (數據同步中)"

# --- 🌍 國際面：美股 S&P 500 連動 ---
def get_international_bias():
    """美股對台股開盤的慣性影響因子"""
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 準確率回測邏輯 (60 日高精度) ---
def calculate_real_accuracy(df, atr_factor, chip_f=1.0, side='high'):
    """回測 60 個交易日的 AI 預估達成率"""
    try:
        df_copy = df.copy().ffill()
        if isinstance(df_copy.columns, pd.MultiIndex): df_copy.columns = df_copy.columns.get_level_values(0)
        backtest_days = min(len(df_copy) - 15, 60)
        if backtest_days <= 0: return 0.0
        hits = 0
        df_copy['ATR'] = (df_copy['High'] - df_copy['Low']).rolling(14).mean()
        
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = df_copy['ATR'].iloc[idx-1]
            if np.isnan(prev_atr): continue
            
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            # 歷史預估點位模擬
            pred_val = prev_close + (prev_atr * atr_factor * chip_f) if side == 'high' else prev_close - (prev_atr * atr_factor / chip_f)
            
            if side == 'high' and actual_val >= pred_val: hits += 1
            elif side == 'low' and actual_val <= pred_val: hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

# --- 獲取股票中文名稱 ---
def get_stock_name(stock_id):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 自動偵測與抓取數據 ---
def fetch_stock_full_data(stock_id, period="150d"):
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        df = yf.download(symbol, period=period, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df, symbol
    return None, None

# --- 🎨 視覺配色組件 (HTML 卡片) ---
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
            <p style="margin-top:10px; font-size:12px; color:#888;">↳ 近 60 日 AI 達成率：<b>{acc:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# 3. 主程式介面邏輯
# =========================================================

# --- A. 首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("目前整合：美股連動、量能慣性、**法人籌碼(FinMind)**、60日高精度回測")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 進入盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 進入深度預估分析", use_container_width=True): navigate_to("forecast")

# --- B. 盤中即時監控 (支援非盤中顯示) ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價監控")
    
    # 時區與盤中判斷
    tw_tz = pytz.timezone('Asia/Taipei')
    now_tw = datetime.datetime.now(tw_tz)
    is_open = now_tw.weekday() < 5 and (datetime.time(9, 0) <= now_tw.time() <= datetime.time(13, 35))

    rt_id = st.text_input("輸入股票代碼以開始監控 (如: 2330):", key="rt_input")
    
    if rt_id:
        with st.spinner('正在獲取數據...'):
            df_rt, sym_rt = fetch_stock_full_data(rt_id, period="5d")
            if df_rt is not None and not df_rt.empty:
                name = get_stock_name(rt_id)
                curr_p = df_rt['Close'].iloc[-1]
                prev_c = df_rt['Close'].iloc[-2]
                
                st.subheader(f"🏠 {name} ({rt_id})")
                if is_open:
                    st.success(f"🟢 盤中交易進行中 (更新：{now_tw.strftime('%H:%M:%S')})")
                else:
                    st.warning(f"🏮 非交易時段 (昨日收盤數據：{df_rt.index[-1].strftime('%Y-%m-%d')})")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("當前成交價", f"{curr_p:.2f}", f"{curr_p-prev_c:+.2f}")
                c2.metric("今日最高", f"{df_rt['High'].iloc[-1]:.2f}")
                c3.metric("今日最低", f"{df_rt['Low'].iloc[-1]:.2f}")

                # 快速 AI 點位
                df_h, _ = fetch_stock_full_data(rt_id, period="100d")
                atr = (df_h['High'] - df_h['Low']).rolling(14).mean().iloc[-1]
                st.divider()
                st.write("🎯 **今日 AI 盤中動態點位參考**")
                st.info(f"建議壓力：{prev_c + (atr * 0.85):.2f} | 建議支撐：{prev_c - (atr * 0.65):.2f}")
            else:
                st.error("❌ 找不到數據")

# --- C. 深度預估分析 ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回首頁"): navigate_to("home")
    st.title("📊 隔日及波段深度預估")
    fc_id = st.text_input("輸入代碼 (例: 2603):", key="fc_input")

    if fc_id:
        with st.spinner('AI 正在計算多因子模型與回測...'):
            df, sym = fetch_stock_full_data(fc_id)
            if df is not None and not df.empty:
                name = get_stock_name(fc_id)
                df = df.ffill()
                
                # 多因子獲取
                market_f, market_pct = get_international_bias()
                chip_f, chip_m = get_chip_factor(fc_id)
                vol_f = 1.05 if df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] else 0.95 
                
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                curr_c = float(df['Close'].iloc[-1])
                total_bias = market_f * chip_f * vol_f
                
                # 點位計算
                ph1 = curr_c + (atr * 0.85 * total_bias)
                ph5 = curr_c + (atr * 1.9 * total_bias)
                pl1 = curr_c - (atr * 0.65 / total_bias)
                pl5 = curr_c - (atr * 1.6 / total_bias)
                
                # 回測準確率
                ah1 = calculate_real_accuracy(df, 0.85, chip_f=chip_f, side='high')
                ah5 = calculate_real_accuracy(df, 1.9, chip_f=chip_f, side='high')
                al1 = calculate_real_accuracy(df, 0.65, chip_f=chip_f, side='low')
                al5 = calculate_real_accuracy(df, 1.6, chip_f=chip_f, side='low')

                # 顯示介面
                st.subheader(f"🏠 {name} ({fc_id})")
                st.write(f"🧬 **{chip_m}**")
                st.write(f"🌍 **美股連動影響**: {market_pct:+.2f}%")
                
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.write("🎯 **壓力預估 (多因子修正)**")
                    stock_box("📈 隔日最高價", ph1, ((ph1/curr_c)-1)*100, ah1, "red")
                    stock_box("🚩 五日最高價", ph5, ((ph5/curr_c)-1)*100, ah5, "red")
                with c2:
                    st.write("🛡️ **支撐預估 (多因子修正)**")
                    stock_box("📉 隔日最低價", pl1, ((pl1/curr_c)-1)*100, al1, "green")
                    stock_box("⚓ 五日最低價", pl5, ((pl5/curr_c)-1)*100, al5, "green")

                # 圖表顯示 (中文化標籤)
                st.divider()
                st.write(f"📈 **{name} 近期價量走勢圖**")
                plot_df = df.tail(40).copy()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="收盤價")
                ax1.axhline(y=ph5, color='#FF4B4B', ls='--', alpha=0.5, label="AI 壓力線")
                ax1.axhline(y=pl5, color='#28A745', ls='--', alpha=0.5, label="AI 支撐線")
                ax1.legend(loc='upper left')
                
                v_colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Close'].iloc[i-1] else 'green' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.7)
                st.pyplot(fig)
            else:
                st.error("❌ 無法抓取歷史數據")
