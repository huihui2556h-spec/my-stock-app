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
# 1. 系統配置與視覺風格 (保持專業深色調，不使用刺眼背景)
# =========================================================
st.set_page_config(page_title="台股 AI 多因子當沖助手 Pro", layout="centered")

# 初始化 session_state 確保頁面切換正常
if 'mode' not in st.session_state:
    st.session_state.mode = "home"

def navigate_to(new_mode):
    st.session_state.mode = new_mode
    st.rerun()

# =========================================================
# 2. 多因子核心運算引擎 (還原您要求的：波動慣性 + 籌碼因子)
# =========================================================

# --- 🎯 籌碼面：FinMind 法人籌碼權重 ---
def get_chip_factor(stock_id):
    """
    從 FinMind 獲取法人買賣超數據。
    邏輯：若法人近五日合計為買超，則給予多頭權重 (1.025)，反之給予空頭權重 (0.975)。
    """
    try:
        from FinMind.data import DataLoader
        dl = DataLoader()
        # 抓取最近 15 天數據以獲得完整的 5 個交易日
        start = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start)
        if not df_inst.empty:
            # 計算三大法人近五日買賣超合計淨額
            net_buy = df_inst.tail(5)['buy'].sum() - df_inst.tail(5)['sell'].sum()
            if net_buy > 0:
                return 1.025, "✅ 籌碼面：法人偏多 (近五日合計買超)"
            else:
                return 0.975, "⚠️ 籌碼面：法人偏空 (近五日合計賣超)"
    except:
        pass
    return 1.0, "ℹ️ 籌碼面：中性 (數據連線中或無數據)"

# --- 🌍 國際面：美股 S&P 500 連動慣性 ---
def get_international_bias():
    """
    抓取美股 S&P 500 指數，計算昨日美股漲跌對今日台股開盤的影響因子。
    """
    try:
        spy = yf.download("^GSPC", period="2d", progress=False)
        if len(spy) < 2: return 1.0, 0.0
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        # 計算漲跌幅百分比
        change = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        # 權重修正：美股每漲 1%，台股權重增加 0.5%
        bias = 1 + (float(change) * 0.5) 
        return bias, float(change) * 100
    except:
        return 1.0, 0.0

# --- 🎯 核心回測函數：60 日高精度達成率 ---
def calculate_real_accuracy(df, atr_factor, chip_f=1.0, side='high'):
    """
    回測過去 60 個交易日，檢查 AI 預估點位是否被實際價格觸及。
    用於計算畫面上的「AI 達成率」。
    """
    try:
        df_copy = df.copy().ffill()
        if isinstance(df_copy.columns, pd.MultiIndex): df_copy.columns = df_copy.columns.get_level_values(0)
        backtest_days = min(len(df_copy) - 15, 60)
        if backtest_days <= 0: return 0.0
        hits = 0
        # 計算波動指標 ATR (14日平均真實波幅)
        df_copy['ATR'] = (df_copy['High'] - df_copy['Low']).rolling(14).mean()
        
        for i in range(1, backtest_days + 1):
            idx = -i
            prev_close = df_copy['Close'].iloc[idx-1]
            prev_atr = df_copy['ATR'].iloc[idx-1]
            if np.isnan(prev_atr): continue
            
            actual_val = df_copy['High'].iloc[idx] if side == 'high' else df_copy['Low'].iloc[idx]
            # 模擬預估公式
            pred_val = prev_close + (prev_atr * atr_factor * chip_f) if side == 'high' else prev_close - (prev_atr * atr_factor / chip_f)
            
            # 判定命中：最高價超越預估高點，或最低價跌破預估低點
            if side == 'high' and actual_val >= pred_val: hits += 1
            elif side == 'low' and actual_val <= pred_val: hits += 1
        return (hits / backtest_days) * 100
    except: return 0.0

# --- 🔍 抓取股票中文名稱 ---
def get_stock_name(stock_id):
    """
    透過 Yahoo 財經爬蟲獲取股票的中文名稱，避免畫面上只有代碼。
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        res = requests.get(url, headers=headers, timeout=5)
        name = re.search(r'<title>(.*?) \(', res.text).group(1)
        return name.split('-')[0].strip()
    except: return f"台股 {stock_id}"

# --- 🛠️ 數據自動抓取：支援上市與上櫃 ---
def fetch_stock_full_data(stock_id, period="150d"):
    """
    自動判定輸入的代碼是上市 (.TW) 還是上櫃 (.TWO)。
    """
    for suffix in [".TW", ".TWO"]:
        symbol = f"{stock_id}{suffix}"
        df = yf.download(symbol, period=period, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df, symbol
    return None, None

# --- 🎨 視覺美化卡片組件 ---
def stock_box(label, price, pct, acc, color_type="red"):
    """
    自定義 HTML 卡片，用於呈現壓力支撐與達成率，維持深色調邊框視覺。
    """
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
# 3. 主程式介面邏輯 (完整中文介面)
# =========================================================

# --- A. 導覽首頁 ---
if st.session_state.mode == "home":
    st.title("⚖️ 台股 AI 多因子交易系統")
    st.write("🔥 系統已整合：美股國際連動、量能慣性、**FinMind 法人籌碼**、60日高精度回測")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚡ 進入盤中即時量價", use_container_width=True): navigate_to("realtime")
    with col_b:
        if st.button("📊 進入深度預估分析", use_container_width=True): navigate_to("forecast")

# --- B. 盤中即時監控頁面 ---
elif st.session_state.mode == "realtime":
    if st.sidebar.button("⬅️ 返回系統首頁"): navigate_to("home")
    st.title("⚡ 盤中即時量價監控")
    
    # 時區判斷 (台灣時間)
    tw_tz = pytz.timezone('Asia/Taipei')
    now_tw = datetime.datetime.now(tw_tz)
    # 判斷是否為台股開盤時間
    is_open = now_tw.weekday() < 5 and (datetime.time(9, 0) <= now_tw.time() <= datetime.time(13, 35))

    rt_id = st.text_input("請輸入股票代碼以開始監控 (如: 2330):", key="rt_input")
    
    if rt_id:
        with st.spinner('正在獲取最新即時數據...'):
            df_rt, _ = fetch_stock_full_data(rt_id, period="5d")
            if df_rt is not None and not df_rt.empty:
                name = get_stock_name(rt_id)
                curr_p = df_rt['Close'].iloc[-1]
                prev_c = df_rt['Close'].iloc[-2]
                
                st.subheader(f"🏠 {name} ({rt_id})")
                if is_open:
                    st.success(f"🟢 盤中交易進行中 (更新時間：{now_tw.strftime('%H:%M:%S')})")
                else:
                    st.warning(f"🏮 目前為非交易時段 (顯示昨日收盤數據：{df_rt.index[-1].strftime('%Y-%m-%d')})")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("當前成交價", f"{curr_p:.2f}", f"{curr_p-prev_c:+.2f}")
                c2.metric("今日最高", f"{df_rt['High'].iloc[-1]:.2f}")
                c3.metric("今日最低", f"{df_rt['Low'].iloc[-1]:.2f}")
                
                # 盤中快速建議區
                st.divider()
                st.write("🎯 **今日 AI 盤中即時壓力支撐參考**")
                df_h, _ = fetch_stock_full_data(rt_id, period="100d")
                atr_rt = (df_h['High'] - df_h['Low']).rolling(14).mean().iloc[-1]
                st.info(f"💡 今日預估壓力：{prev_c + (atr_rt * 0.85):.2f} | 今日預估支撐：{prev_c - (atr_rt * 0.65):.2f}")
            else:
                st.error("❌ 找不到該代碼數據，請檢查輸入。")

# --- C. 深度預估分析頁面 (含買賣點位) ---
elif st.session_state.mode == "forecast":
    if st.sidebar.button("⬅️ 返回系統首頁"): navigate_to("home")
    st.title("📊 隔日及波段預估深度分析")
    fc_id = st.text_input("請輸入股票代碼 (例: 2603):", key="fc_input")

    if fc_id:
        with st.spinner('AI 正在跑多因子模型與 60 日回測...'):
            df, _ = fetch_stock_full_data(fc_id)
            if df is not None and not df.empty:
                name = get_stock_name(fc_id)
                df = df.ffill()
                
                # 1. 因子獲取
                market_f, market_pct = get_international_bias()
                chip_f, chip_m = get_chip_factor(fc_id)
                # 量能因子：今日成交量與 5 日均量關係
                vol_f = 1.05 if df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] else 0.95 
                
                # 2. 核心計算
                atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                curr_c = float(df['Close'].iloc[-1])
                # 綜合加權因子
                total_bias = market_f * chip_f * vol_f
                
                # 3. 預估點位計算
                ph1 = curr_c + (atr * 0.85 * total_bias) # 隔日最高
                ph5 = curr_c + (atr * 1.9 * total_bias)  # 五日最高
                pl1 = curr_c - (atr * 0.65 / total_bias) # 隔日最低
                pl5 = curr_c - (atr * 1.6 / total_bias)  # 五日最低
                
                # 4. 回測準確率
                ah1 = calculate_real_accuracy(df, 0.85, chip_f=chip_f, side='high')
                ah5 = calculate_real_accuracy(df, 1.9, chip_f=chip_f, side='high')
                al1 = calculate_real_accuracy(df, 0.65, chip_f=chip_f, side='low')
                al5 = calculate_real_accuracy(df, 1.6, chip_f=chip_f, side='low')

                # --- 介面呈現 (中文名稱顯示) ---
                st.subheader(f"🏠 {name} ({fc_id})")
                st.write(f"🧬 **{chip_m}**")
                st.write(f"🌍 **國際局勢參考 (美股漲跌)**: {market_pct:+.2f}%")
                
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.write("🎯 **壓力預估 (多因子修正)**")
                    stock_box("📈 隔日預估最高", ph1, ((ph1/curr_c)-1)*100, ah1, "red")
                    stock_box("🚩 五日波段最高", ph5, ((ph5/curr_c)-1)*100, ah5, "red")
                with c2:
                    st.write("🛡️ **支撐預估 (多因子修正)**")
                    stock_box("📉 隔日預估最低", pl1, ((pl1/curr_c)-1)*100, al1, "green")
                    stock_box("⚓ 五日波段最低", pl5, ((pl5/curr_c)-1)*100, al5, "green")

                # --- 🏹 這裡是你最在意的：明日當沖建議價格 (補回！) ---
                st.divider()
                st.markdown("### 🏹 明日當沖建議參考點位 (AI 核心策略)")
                d1, d2, d3 = st.columns(3)
                # 強勢追多：考慮量能因子修正後的買入點
                d1.info(f"🔹 **強勢追多買點**\n\n**{curr_c + (atr * 0.1 * vol_f):.2f}**")
                # 低接買點：考慮支撐位與美股修正
                d2.error(f"🔹 **回測支撐低接**\n\n**{curr_c - (atr * 0.45 / market_f):.2f}**")
                # 短線獲利：目標賣點
                d3.success(f"🔸 **短線分批停利**\n\n**{curr_c + (atr * 0.75 * total_bias):.2f}**")

                # --- 📈 價量彩色圖表 ---
                st.divider()
                st.write(f"📊 **{name} 近期價量走勢與 AI 點位圖**")
                plot_df = df.tail(40).copy()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                # 價格線
                ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label="收盤價")
                ax1.axhline(y=ph5, color='#FF4B4B', ls='--', alpha=0.5, label="AI 壓力線")
                ax1.axhline(y=pl5, color='#28A745', ls='--', alpha=0.5, label="AI 支撐線")
                ax1.set_ylabel("價格 (TWD)")
                ax1.legend(loc='upper left')
                
                # 彩色成交量 (紅漲綠跌)
                v_colors = ['red' if plot_df['Close'].iloc[i] >= plot_df['Close'].iloc[i-1] else 'green' for i in range(len(plot_df))]
                ax2.bar(plot_df.index, plot_df['Volume'], color=v_colors, alpha=0.7)
                ax2.set_ylabel("成交量")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                st.info("📘 **圖表說明**：上方為收盤價走勢；下方為成交量（紅漲綠跌）。虛線為 AI 預判之波段壓力與支撐。")
            else:
                st.error("❌ 無法抓取歷史數據，請確認股票代碼是否正確。")
