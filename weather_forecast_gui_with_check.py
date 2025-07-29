
import streamlit as st
import subprocess
import time
import threading

st.set_page_config(layout="wide")
st.title("Climate Crystal Orb")

# -------------------- Bildpfade --------------------
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
    "images/loading/loading15.png"
    ]

image_path_merged = [
    "images/merging/merging1.png",
    "images/merging/merging2.png",
    "images/merging/merging3.png",
    "images/merging/merging4.png",
    "images/merging/merging5.png",
    "images/merging/merging1.png",
    "images/merging/merging2.png",
    "images/merging/merging3.png",
    "images/merging/merging4.png",
    "images/merging/merging5.png",
    "images/merging/merging1.png",
    "images/merging/merging2.png",
    "images/merging/merging3.png",
    "images/merging/merging4.png",
    "images/merging/merging5.png"
]

# -------------------- Container vorbereiten --------------------
output_container = st.empty()
image_container = st.empty()
button_container = st.empty()
progress_bar = st.empty()

# -------------------- Session Init --------------------
if "phase" not in st.session_state:
    st.session_state.phase = 1
if "cloud_result" not in st.session_state:
    st.session_state.cloud_result = {}
if "merge_result" not in st.session_state:
    st.session_state.merge_result = {}

# -------------------- Skriptlauf-Funktionen --------------------
def run_script(script_path: str, result_container: dict, capture_output=True):
    proc = subprocess.Popen(
        ["python3", script_path],
        stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if capture_output else subprocess.DEVNULL,
        text=True
    )
    output = ""
    if capture_output:
        for line in proc.stdout:
            output += line
        result_container["stdout"] = output
    proc.wait()

# -------------------- Fortschrittsanzeige --------------------
def show_progress_loop(thread, image_list, text="Verarbeite ..."):
    idx = 0
    steps = 100
    while thread.is_alive():
        img = image_list[idx % len(image_list)]
        image_container.image(img, use_container_width=True)
        progress_bar.progress(min(idx % steps / steps, 1.0), text=text)
        time.sleep(1.2)
        idx += 1
    progress_bar.empty()

# -------------------- Anzeige Phase 1 --------------------
if st.session_state.phase == 1:
    if button_container.button("▶️ Starte cloud_utils.py"):
        result = {}
        thread = threading.Thread(target=run_script, args=("utils/cloud_utils.py", result))
        thread.start()
        show_progress_loop(thread, image_path_utils, "cloud_utils.py läuft ...")
        thread.join()

        st.session_state.cloud_result = result
        st.session_state.phase = 2
        button_container.empty()
        image_container.empty()
        output_container.success("✅ cloud_utils.py abgeschlossen")
        output_container.code(result.get("stdout", "").strip().split("\n")[-1] or "Keine Ausgabe")

# -------------------- Anzeige Phase 2 --------------------
if st.session_state.phase == 2:
    if button_container.button("🚀 Starte merge_tiles_with_cache.py + Prüfung"):
        merge_result = {}
        thread = threading.Thread(target=run_script, args=("data/preprocessing/merge_tiles_with_cache.py", merge_result, False))
        thread.start()
        show_progress_loop(thread, image_path_merged, "Merging läuft ...")
        thread.join()

        st.session_state.merge_result = merge_result
        button_container.empty()
        progress_bar.empty()  # explizit ausblenden

        # Sofortiger Start von check_missing_merged_tiles.py
        output_container.info("🔍 Starte automatische Prüfung ...")
        check_result = {}
        check_thread = threading.Thread(target=run_script, args=("data/preprocessing/check_missing_merged_tiles.py", check_result))
        check_thread.start()
        show_progress_loop(check_thread, image_path_merged, "Prüfe auf fehlende Kacheln ...")
        check_thread.join()

        output_container.success("✅ Prüfung abgeschlossen")
        output_container.code(check_result.get("stdout", "") or "Keine Ausgabe")
