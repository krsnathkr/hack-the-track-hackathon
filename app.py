import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
from src.data_processor import load_and_pivot_data
from src.overtake_model import RaceAnalysis

st.set_page_config(page_title="Toyota GR Cup - Overtake AI", layout="wide")

# --- SETUP ---


@st.cache_data
def get_race_data():
    # CHANGE THIS PATH TO YOUR ACTUAL FILE
    full_path = "data/barber/R1_barber_telemetry_data.csv"
    try:
        # Try to load the full file
        with open(full_path, 'r') as f:
            pass # Just check if it exists/can be opened
        filepath = full_path
    except FileNotFoundError:
        # Fallback to sample data for deployment
        filepath = "data/deployment_sample.csv"
        st.warning("⚠️ Using deployment sample data (Full telemetry file not found).")
            
    return load_and_pivot_data(filepath)

try:
    df = get_race_data()
    race_model = RaceAnalysis(df)
    vehicle_ids = list(race_model.vehicle_ids)
    
    # Get Track Map Background (Static)
    track_bg = df[['VBOX_Long_Min', 'VBOX_Lat_Min']].iloc[::50].drop_duplicates()
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🏎️ Race Engineer AI")



# Determine default indices based on demo data
hero_index = 0


hero_car = st.sidebar.selectbox("Hero Car", vehicle_ids, index=hero_index)

# Logic: Remove Hero from Rival list
rival_options = [v for v in vehicle_ids if v != hero_car]

# Determine default rival index
rival_index = 0


rival_car = st.sidebar.selectbox("Rival Car", rival_options, index=rival_index)

mode = st.sidebar.radio("Rival Mode", ["Live Battle", "Ghost (Session Best)"])

# Determine Rival Lap
rival_target_lap = None

if mode == "Ghost (Session Best)":
    ghost_vid, ghost_lap = race_model.get_fastest_lap()
    if ghost_vid:
        rival_car = ghost_vid
        rival_target_lap = ghost_lap
        st.sidebar.success(f"👻 Ghost: {ghost_vid} (Lap {ghost_lap})")
    else:
        st.sidebar.error("No Ghost Lap found.")

# --- SIMULATION STATE ---
if 'sim_step' not in st.session_state:
    st.session_state.sim_step = 0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'smoothed_prob' not in st.session_state:
    st.session_state.smoothed_prob = 0.0

start_btn = st.sidebar.button("▶️ Start Simulation")
stop_btn = st.sidebar.button("⏸️ Pause")
reset_btn = st.sidebar.button("🔄 Reset")

if start_btn: st.session_state.running = True
if stop_btn: st.session_state.running = False
if reset_btn: 
    st.session_state.sim_step = 0
    st.session_state.running = False

# Speed Control
speed_multiplier = st.sidebar.slider("Simulation Speed", 1, 10, 2)



# --- DATA PREP FOR LOOP ---
# 1. Get Hero Timeline (Full Session)
hero_full = race_model.vehicle_data[hero_car]

# 2. Get Rival Timeline (Target Lap ONLY)
# This optimization prevents lag inside the loop
rival_full = race_model.vehicle_data[rival_car]
if rival_target_lap:
    rival_lap_data = rival_full[rival_full['Lap_Number'] == rival_target_lap].sort_values('Laptrigger_lapdist_dls')
else:
    # If "Live Battle", just grab the same lap number as the hero currently is?
    # For simplicity, we'll just use the Rival's Lap 1 data if no specific lap selected
    # Or better: Use the whole dataset and filter dynamically (slower)
    rival_lap_data = rival_full 

# --- MAIN DASHBOARD ---
# Custom CSS for Bento Box styling
st.markdown("""
<style>
    .metric-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #aaaaaa;
    }
    .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }
    .factor-badge {
        background-color: #00F0FF;
        color: #000000;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 5px;
        display: inline-block;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Placeholder for the entire dashboard
dashboard_placeholder = st.empty()

def render_dashboard(sim_step):
    # Get Current Hero State
    curr_idx = min(sim_step, len(hero_full)-1)
    hero_now = hero_full.iloc[curr_idx]
    current_dist = hero_now['Laptrigger_lapdist_dls']
    hero_lap = hero_now['Lap_Number']

    # Get Rival State (Synchronized by DISTANCE)
    target_lap = rival_target_lap
    if mode == "Live Battle":
        target_lap = hero_lap 

    rival_now = race_model.get_car_state_at_distance(rival_car, current_dist, target_lap=target_lap)

    # Calculate Math
    prob, reasons, metrics, feedback = race_model.calculate_overtake_probability(hero_now, rival_now)

    # --- SMOOTHING ---
    # Alpha: 0.1 = very smooth/slow, 0.9 = very reactive/jittery
    alpha = 0.15 
    st.session_state.smoothed_prob = (alpha * prob) + ((1 - alpha) * st.session_state.smoothed_prob)
    
    display_prob = st.session_state.smoothed_prob

    # Prepare Data for Charts (Last 500m)
    dist_min = current_dist - 500
    dist_max = current_dist

    # Filter Hero History
    h_hist = hero_full[(hero_full['Laptrigger_lapdist_dls'] > dist_min) & (hero_full['Laptrigger_lapdist_dls'] <= dist_max) & (hero_full['Lap_Number'] == hero_lap)]
    h_hist['Car'] = 'Hero'

    # Filter Rival History
    if target_lap:
        r_hist = rival_full[(rival_full['Lap_Number'] == target_lap) & 
                            (rival_full['Laptrigger_lapdist_dls'] > dist_min) & 
                            (rival_full['Laptrigger_lapdist_dls'] <= dist_max)]
    else:
        r_hist = rival_lap_data[(rival_lap_data['Laptrigger_lapdist_dls'] > dist_min) & 
                                (rival_lap_data['Laptrigger_lapdist_dls'] <= dist_max)]

    r_hist['Car'] = 'Rival'

    # Optimization: Downsample data for rendering (every 5th point)
    # This significantly reduces the JSON payload sent to the frontend
    h_hist_lite = h_hist.iloc[::5]
    r_hist_lite = r_hist.iloc[::5]

    chart_data = pd.concat([h_hist_lite, r_hist_lite])

    with dashboard_placeholder.container():
        st.markdown("### 🏁 Race Control")

        # ROW 1: HEADS UP DISPLAY (Probability | Map | G-G)
        col_prob, col_map, col_gg = st.columns([1, 2, 1])

        with col_prob:
            st.markdown("#### Overtake Chance")
            # Probability Gauge
            color = "#00FF00" if display_prob > 60 else "#FFFF00" if display_prob > 30 else "#FF0055"
            st.markdown(f"""
                <div style="text-align:center; background-color:#1E1E1E; padding:20px; border-radius:10px; border: 2px solid {color}; margin-bottom: 10px;">
                    <h1 style="font-size:48px; color:{color}; margin:0;">{display_prob:.0f}%</h1>
                    <p style="color:#aaaaaa; margin:0;">Probability</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Key Decision Factors
            if reasons:
                st.markdown("**Key Factors:**")
                html_content = ""
                for r in reasons:
                    html_content += f'<div class="factor-badge" style="margin-bottom:5px;">{r}</div>'
                st.markdown(html_content, unsafe_allow_html=True)

        with col_map:
            st.markdown("#### 🗺️ Live Track Map")
            # Track Map with Dots
            base = alt.Chart(track_bg).mark_circle(size=2, color='gray').encode(
                x=alt.X('VBOX_Long_Min', axis=None, scale=alt.Scale(zero=False)),
                y=alt.Y('VBOX_Lat_Min', axis=None, scale=alt.Scale(zero=False))
            )
            
            h_dot = alt.Chart(pd.DataFrame([hero_now])).mark_circle(size=100, color='#00F0FF').encode(
                x='VBOX_Long_Min', y='VBOX_Lat_Min', tooltip=['Speed']
            )
            
            dots = h_dot
            if rival_now is not None:
                r_dot = alt.Chart(pd.DataFrame([rival_now])).mark_circle(size=100, color='#FF0055').encode(
                    x='VBOX_Long_Min', y='VBOX_Lat_Min', tooltip=['Speed']
                )
                dots = h_dot + r_dot
                
            st.altair_chart(base + dots, use_container_width=True)

        with col_gg:
            st.markdown("#### 📉 G-Force Traces")
            
            # 1. Longitudinal G (Accel/Brake)
            long_g_chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X('Laptrigger_lapdist_dls', title=None, axis=None, scale=alt.Scale(domain=[dist_min, dist_max])),
                y=alt.Y('accx_can', title='Long G', scale=alt.Scale(domain=[-2, 2])),
                color=alt.Color('Car', legend=None, scale=alt.Scale(domain=['Hero', 'Rival'], range=['#00F0FF', '#FF0055'])),
                tooltip=['Speed', 'accx_can']
            ).properties(height=120)

            # 2. Lateral G (Cornering)
            lat_g_chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X('Laptrigger_lapdist_dls', title='Distance (m)', scale=alt.Scale(domain=[dist_min, dist_max])),
                y=alt.Y('accy_can', title='Lat G', scale=alt.Scale(domain=[-2, 2])),
                color=alt.Color('Car', legend=None, scale=alt.Scale(domain=['Hero', 'Rival'], range=['#00F0FF', '#FF0055'])),
                tooltip=['Speed', 'accy_can']
            ).properties(height=120)
            
            st.altair_chart(long_g_chart & lat_g_chart, use_container_width=True)


        # ROW 2: MAIN TELEMETRY (Speed)
        st.markdown("### 📈 Speed Trace")
        # Altair Speed Chart
        speed_chart = alt.Chart(chart_data).mark_line().encode(
            x=alt.X('Laptrigger_lapdist_dls', title='Distance (m)', scale=alt.Scale(domain=[dist_min, dist_max])),
            y=alt.Y('Speed', title='Speed (km/h)', scale=alt.Scale(zero=False)),
            color=alt.Color('Car', scale=alt.Scale(domain=['Hero', 'Rival'], range=['#00F0FF', '#FF0055'])),
            tooltip=['Speed', 'aps', 'pbrake_f']
        ).properties(height=250)

        st.altair_chart(speed_chart, use_container_width=True)


        # ROW 3: DEEP DIVE (Inputs | Engineer)
        col_inputs, col_eng = st.columns([1, 1])

        with col_inputs:
            st.markdown("### 🦶 Driver Inputs")
            
            # Throttle (aps)
            throttle_chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X('Laptrigger_lapdist_dls', title=None, axis=None, scale=alt.Scale(domain=[dist_min, dist_max])),
                y=alt.Y('aps', title='Throttle %', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('Car', legend=None, scale=alt.Scale(domain=['Hero', 'Rival'], range=['#00F0FF', '#FF0055'])),
            ).properties(height=150)
            
            # Brake (pbrake_f)
            brake_chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X('Laptrigger_lapdist_dls', title='Distance (m)', scale=alt.Scale(domain=[dist_min, dist_max])),
                y=alt.Y('pbrake_f', title='Brake (bar)'),
                color=alt.Color('Car', legend=None, scale=alt.Scale(domain=['Hero', 'Rival'], range=['#00F0FF', '#FF0055'])),
            ).properties(height=150)
            
            st.altair_chart(throttle_chart & brake_chart, use_container_width=True)

        with col_eng:
            st.markdown("### 📻 Race Engineer Station")
            
            # 1. Live Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Gap", f"{metrics['gap']:.1f} m")
            with m_col2:
                st.metric("Closing", f"{metrics['closing_speed']:.1f} m/s")
            with m_col3:
                ttc_str = f"{metrics['time_to_close']:.1f}s" if metrics['time_to_close'] != float('inf') else "∞"
                st.metric("TTC", ttc_str)
                
            st.divider()
            
            # 2. Sector Analysis
            st.markdown("**⏱️ Sector Deltas**")
            
            # Optimization: Cache sector analysis
            # Create a unique key for the current lap pair
            sector_key = f"{hero_car}_{hero_lap}_{rival_car}_{target_lap}"
            
            if 'sector_cache' not in st.session_state:
                st.session_state.sector_cache = {}
                
            if sector_key not in st.session_state.sector_cache:
                if target_lap:
                    h_lap_full = hero_full[hero_full['Lap_Number'] == hero_lap]
                    r_lap_full = rival_full[rival_full['Lap_Number'] == target_lap]
                    st.session_state.sector_cache[sector_key] = race_model.analyze_sectors(h_lap_full, r_lap_full)
                else:
                    st.session_state.sector_cache[sector_key] = None
            
            sectors = st.session_state.sector_cache[sector_key]
            
            if sectors:
                sec_df = pd.DataFrame(sectors)
                # Format for display
                sec_df['Delta'] = sec_df['delta'].apply(lambda x: f"{x:+.3f}s")
                sec_df['Winner'] = sec_df['delta'].apply(lambda x: "HERO" if x < 0 else "RIVAL")
                st.dataframe(sec_df[['sector', 'Delta', 'Winner']], hide_index=True, use_container_width=True)

            st.divider()

            # 3. AI Feedback
            st.markdown("**🎙️ Radio Messages**")
            if feedback:
                for f in feedback:
                    st.info(f"{f}")
            else:
                st.caption("No critical messages.")
                
            st.divider()

# --- MAIN LOOP ---
if st.session_state.running:
    # Clear any existing static content before starting animation
    dashboard_placeholder.empty()
    
    # Continuous loop for smooth animation
    while st.session_state.running:
        # Update Step
        st.session_state.sim_step += speed_multiplier
        if st.session_state.sim_step >= len(hero_full):
            st.session_state.sim_step = 0
            
        # Render
        render_dashboard(st.session_state.sim_step)
        
        # Speed Control
        time.sleep(0.02)
else:
    # Static Render
    render_dashboard(st.session_state.sim_step)