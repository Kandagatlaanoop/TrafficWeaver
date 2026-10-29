# command_center_dashboard.py
import streamlit as st
import numpy as np
import traci
import sumolib
import time
from datetime import datetime
import mss
import cv2
import atexit
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import pygetwindow as gw

# --- NEW: Path Correction ---
# This ensures that Python can find the 'marl' module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from marl.marl_env import TrafficEnv # Corrected import path

# --- Page Configuration ---
st.set_page_config(page_title="TrafficWeaver Command Center", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
    <style>
        .stMetric { border-radius: 10px; padding: 15px; background-color: #262730; border: 1px solid #4F8BF9; }
    </style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'event_log' not in st.session_state:
    st.session_state.event_log = ["System Initialized."]

# --- Helper Functions ---
def add_log(message):
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, f"**[{now}]** {message}")
    if len(st.session_state.event_log) > 10: st.session_state.event_log.pop()

# --- Ray and Model Loading ---
@st.cache_resource
def load_hive_mind_model(checkpoint_path):
    try:
        ray.init(logging_level="ERROR", ignore_reinit_error=True)
        
        config = (
            PPOConfig()
            .environment(TrafficEnv)
            .framework("torch")
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
            )
            .resources(num_gpus=0)
            .env_runners(num_env_runners=0)
        )
        trainer = config.build()
        trainer.restore(checkpoint_path)
        add_log("Hive Mind AI model loaded successfully.")
        return trainer
    except Exception as e:
        st.error(f"Could not load Ray/RLlib model: {e}")
        return None

# --- SUMO Connection Management ---
def start_sumo():
    if not traci.isLoaded():
        try:
            sumo_binary = sumolib.checkBinary('sumo-gui')
            sumo_cmd = [sumo_binary, "-c", "simulation/2x2_grid.sumocfg", "--no-warnings", "true", "--time-to-teleport", "-1"]
            traci.start(sumo_cmd, label="sumo_dashboard")
            add_log("SUMO simulation connected.")
        except Exception as e:
            st.error(f"Could not start SUMO: {e}")
            st.stop()

def close_sumo():
    if traci.isLoaded():
        traci.close()

atexit.register(close_sumo)

# --- Main Application ---
st.title("🌐 TrafficWeaver Command Center") # Replaced emoji

checkpoint_path = st.text_input("Enter the path to your trained model checkpoint:", "")

hive_mind = None
if checkpoint_path:
    hive_mind = load_hive_mind_model(checkpoint_path)

live_feed_col, control_col = st.columns([2, 1])

with live_feed_col:
    st.header("🖥️ Live Simulation Feed") # Replaced emoji
    FRAME_WINDOW = st.image([])

with control_col:
    analytics_tab, log_tab = st.tabs(["📊 Analytics", "📋 Event Log"]) # Replaced emojis
    with analytics_tab:
        st.subheader("System Status")
        status_placeholder = st.empty()
        st.subheader("Live Grid KPIs")
        kpi1, kpi2 = st.columns(2)
        total_cars_placeholder = kpi1.empty()
        total_wait_placeholder = kpi2.empty()
        st.subheader("Agent Status")
        agent_status_placeholder = st.empty()
    with log_tab:
        st.subheader("Dynamic Event Log")
        log_placeholder = st.empty()

run_simulation = st.checkbox("Run Simulation", value=True)

if run_simulation and hive_mind:
    start_sumo()
    # This is now guaranteed to work because the environment won't close the connection
    conn = traci.getConnection("sumo_dashboard")
    env = TrafficEnv()
    
    observations, infos = env.reset()
    
    with mss.mss() as sct:
        try:
            sumo_window_list = gw.getWindowsWithTitle('SUMO-GUI')
            if not sumo_window_list:
                st.warning("SUMO GUI window not found. Screen capture will be disabled.")
                sumo_window = None
            else:
                sumo_window = sumo_window_list[0]
                monitor = {"top": sumo_window.top, "left": sumo_window.left, "width": sumo_window.width, "height": sumo_window.height}
        except Exception:
            st.warning("Could not attach to SUMO window for screen capture.")
            sumo_window = None

        while run_simulation:
            try:
                actions = hive_mind.compute_actions(observations)
                observations, reward, terminated, truncated, info = env.step(actions)
                
                status_placeholder.success("💡 HIVE MIND CONTROL ACTIVE", icon="✅") # Replaced emoji

                # Update KPIs
                total_cars = conn.simulation.getDepartedNumber() - conn.simulation.getArrivedNumber()
                current_state = env._get_state()
                total_waiting = sum(state.sum() for state in current_state.values())

                total_cars_placeholder.metric("Live Vehicle Count", int(total_cars))
                total_wait_placeholder.metric("Total Waiting Cars", int(total_waiting))

                # Update Agent Status
                agent_statuses = []
                for agent_id in env.agents:
                    phase = conn.trafficlight.getPhase(agent_id)
                    phase_name = "N-S Green" if phase in [0, 1] else "E-W Green"
                    agent_statuses.append(f"**{agent_id}:** {phase_name}")
                agent_status_placeholder.markdown("\n\n".join(agent_statuses))

                log_placeholder.info("\n".join(st.session_state.get('event_log', [])))

                # Update Live Feed
                if sumo_window:
                    img = np.array(sct.grab(monitor))
                    FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

                if terminated.get("__all__") or truncated.get("__all__"):
                    add_log("Episode finished. Resetting simulation.")
                    observations, infos = env.reset()

                time.sleep(0.1)

            except traci.TraCIException:
                add_log("SUMO window was closed. Stopping demo.")
                break
else:
    close_sumo()
    st.info("Simulation stopped or AI model not loaded.")