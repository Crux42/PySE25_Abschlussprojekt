
import streamlit as st
import subprocess
import time
import threading

st.set_page_config(layout="wide")
st.title("☁️ Wetterprozess-Steuerung")

# -------------------- Bilder definieren --------------------
image_path_utils = [
    "images/loading/loading1.png",
    "images/loading/loading2.png",
    "images/loading/loading3.png", 
    "images/loading/loading4.png",
    "images/loading/loading5.png",
    "images/loading/loading6.png",
    "images/loading/loading7.png",
    "images/loading/loading8.png",
    "images/loading/loading9.png",
    "images/loading/loading10.png",
    "images/loading/loading11.png",
    "images/loading/loading12.png",
    "images/loading/loading13.png",
    "images/loading/loading14.png",
    "images/loading/loading15.png",
]

image_path_merged = [
    "images/merging/merging1.png",
    "images/merging/merging2.png",
    "images/merging/merging3.png",
    "images/merging/merging4.png",
    "images/merging/merging5.png",
]

# -------------------- Container vorbereiten --------------------
output_container = st.empty()
image_container = st.empty()
button_container = st.empty()
spinner_container = st.empty()

# -------------------- Session Init --------------------
if "phase" not in st.session_state:
    st.session_state.phase = 1
if "thread_result" not in st.session_state:
    st.session_state.thread_result = {}

# -------------------- Phase 1 Funktion --------------------
def run_cloud_utils():
    proc = subprocess.Popen(
        ["python3", "utils/cloud_utils.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate()
    st.session_state.thread_result["cloud_utils"] = {
        "stdout": stdout,
        "stderr": stderr
    }
    st.session_state.phase = 2

# -------------------- Phase 2 Funktion --------------------
def run_merge_tiles():
    proc = subprocess.Popen(
        ["python3", "preprocessing/merge_tiles_with_cache.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate()
    st.session_state.thread_result["merge_tiles"] = {
        "stdout": stdout,
        "stderr": stderr
    }

# -------------------- Anzeige Phase 1 --------------------
if st.session_state.phase == 1:
    if button_container.button("▶️ Starte cloud_utils.py"):
        thread = threading.Thread(target=run_cloud_utils)
        thread.start()

        with spinner_container.status("⏳ Lade Daten mit cloud_utils.py...", expanded=False):
            idx = 0
            while thread.is_alive():
                img = image_path_utils[idx % len(image_path_utils)]
                image_container.image(img, use_container_width=True)
                time.sleep(1.2)
                idx += 1
        button_container.empty()
        image_container.empty()

        result = st.session_state.thread_result.get("cloud_utils")
        if result:
            output_container.success("✅ cloud_utils.py abgeschlossen")
            output_container.code(result["stdout"].strip().split("\n")[-1] if result["stdout"] else "Keine Ausgabe")
            if result["stderr"]:
                output_container.error(result["stderr"])

# -------------------- Anzeige Phase 2 --------------------
if st.session_state.phase == 2:
    if button_container.button("🚀 Starte merge_tiles_with_cache.py"):
        thread = threading.Thread(target=run_merge_tiles)
        thread.start()

        with spinner_container.status("⏳ Merging läuft...", expanded=False):
            idx = 0
            while thread.is_alive():
                img = image_path_merged[idx % len(image_path_merged)]
                image_container.image(img, use_container_width=True)
                time.sleep(1.2)
                idx += 1
        button_container.empty()
        image_container.empty()

        result = st.session_state.thread_result.get("merge_tiles")
        if result:
            output_container.success("✅ merge_tiles_with_cache.py abgeschlossen")
            output_container.code(result["stdout"].strip().split("\n")[-1] if result["stdout"] else "Keine Ausgabe")
            if result["stderr"]:
                output_container.error(result["stderr"])
