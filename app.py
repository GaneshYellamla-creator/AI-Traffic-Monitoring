import os
import tempfile
from collections import defaultdict
import cv2
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# OPTIONAL RAG + LLM IMPORTS
# ---------------------------------------------------------

try:
    from llm.assistant import analyze_event
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Traffic Monitor",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# LOAD LOGO
# ---------------------------------------------------------

LOGO_PATH = "assets/logo.png"

if os.path.exists(LOGO_PATH):
    logo = Image.open(LOGO_PATH)
else:
    logo = None


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 10px;
    }

    .metric-title {
        font-size: 14px;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .alert-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,80,80,0.5);
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    if logo is not None:
        st.image(
            logo,
            use_container_width=True
        )

    st.markdown("---")

    st.title("AI Traffic Monitor")

    st.caption(
        "Detect • Track • Analyze • Safer Roads"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Video Analysis",
            "AI Traffic Assistant",
            "About Project"
        ]
    )

    st.markdown("---")

    st.success("System Ready")

    st.caption(
        "YOLO + Object Tracking + RAG + LLM"
    )


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "processed_data" not in st.session_state:
    st.session_state.processed_data = []

if "last_event" not in st.session_state:
    st.session_state.last_event = None

if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0

if "processed_video" not in st.session_state:
    st.session_state.processed_video = False


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    if logo is not None:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                logo,
                use_container_width=True
            )

    st.markdown(
        '<div class="main-title">'
        '🚦 AI Smart Traffic & Road Safety Monitor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent object detection, tracking, traffic analytics, '
        'RAG-based rule retrieval and LLM assistance'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Metrics
    data = st.session_state.processed_data

    total_objects = len(
        set(
            item["id"]
            for item in data
            if item["id"] is not None
        )
    )

    cars = len(
        set(
            item["id"]
            for item in data
            if item["class"] == "car"
            and item["id"] is not None
        )
    )

    motorcycles = len(
        set(
            item["id"]
            for item in data
            if item["class"] == "motorcycle"
            and item["id"] is not None
        )
    )

    people = len(
        set(
            item["id"]
            for item in data
            if item["class"] == "person"
            and item["id"] is not None
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Tracked Objects",
            total_objects
        )

    with col2:
        st.metric(
            "Cars",
            cars
        )

    with col3:
        st.metric(
            "Motorcycles",
            motorcycles
        )

    with col4:
        st.metric(
            "People",
            people
        )

    st.divider()

    st.subheader("📊 Traffic Overview")

    if data:

        df = pd.DataFrame(data)

        class_counts = (
            df.groupby("class")["id"]
            .nunique()
            .reset_index()
        )

        class_counts.columns = [
            "Object Type",
            "Count"
        ]

        st.bar_chart(
            class_counts.set_index("Object Type")
        )

        st.subheader("Recent Detection Data")

        st.dataframe(
            df.tail(20),
            use_container_width=True
        )

    else:

        st.info(
            "No video has been analyzed yet. "
            "Go to Video Analysis and upload a traffic video."
        )


# =========================================================
# VIDEO ANALYSIS
# =========================================================

elif page == "Video Analysis":

    st.title("🎥 Traffic Video Analysis")

    st.write(
        "Upload a traffic video and the AI system will detect "
        "and track objects using YOLO."
    )

    uploaded_file = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:

        # Save uploaded video temporarily
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(
                uploaded_file.name
            )[1]
        )

        temp_file.write(
            uploaded_file.read()
        )

        video_path = temp_file.name

        st.success(
            f"Video loaded: {uploaded_file.name}"
        )

        # -------------------------------------------------
        # IMPORT YOLO
        # -------------------------------------------------

        try:
            from ultralytics import YOLO
        except ImportError:

            st.error(
                "Ultralytics is not installed."
            )

            st.code(
                "pip install ultralytics"
            )

            st.stop()

        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        MODEL_PATH = "models/yolov8n.pt"

        if not os.path.exists(MODEL_PATH):

            st.warning(
                "YOLO model not found in models/."
            )

            st.info(
                "The Ultralytics model will be downloaded "
                "automatically the first time you run it."
            )

            model = YOLO("yolov8n.pt")

        else:

            model = YOLO(MODEL_PATH)

        # -------------------------------------------------
        # PROCESS BUTTON
        # -------------------------------------------------

        if st.button(
            "▶ Start AI Traffic Analysis",
            type="primary",
            use_container_width=True
        ):

            st.session_state.processed_data = []
            st.session_state.last_event = None

            cap = cv2.VideoCapture(
                video_path
            )

            if not cap.isOpened():

                st.error(
                    "Unable to open the uploaded video."
                )

                st.stop()

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:
                fps = 25

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            # -------------------------------------------------
            # OUTPUT VIDEO
            # -------------------------------------------------

            output_path = os.path.join(
                tempfile.gettempdir(),
                "ai_traffic_output.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # -------------------------------------------------
            # STREAMLIT UI
            # -------------------------------------------------

            video_placeholder = st.empty()

            progress = st.progress(0)

            status = st.empty()

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            frame_number = 0

            # Store previous positions
            previous_positions = {}

            # Store unique objects
            seen_objects = set()

            # YOLO classes we care about
            useful_classes = {
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "truck"
            }

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                # -------------------------------------------------
                # YOLO TRACKING
                # -------------------------------------------------

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                result = results[0]

                annotated_frame = result.plot()

                # -------------------------------------------------
                # READ DETECTIONS
                # -------------------------------------------------

                if result.boxes is not None:

                    boxes = result.boxes

                    for i in range(
                        len(boxes)
                    ):

                        class_id = int(
                            boxes.cls[i]
                        )

                        class_name = model.names[
                            class_id
                        ]

                        if class_name not in useful_classes:
                            continue

                        confidence = float(
                            boxes.conf[i]
                        )

                        # Tracking ID
                        track_id = None

                        if boxes.id is not None:

                            track_id = int(
                                boxes.id[i]
                            )

                        # Bounding box
                        x1, y1, x2, y2 = map(
                            int,
                            boxes.xyxy[i].tolist()
                        )

                        center_x = int(
                            (x1 + x2) / 2
                        )

                        center_y = int(
                            (y1 + y2) / 2
                        )

                        # -------------------------------------------------
                        # DIRECTION ESTIMATION
                        # -------------------------------------------------

                        direction = "Unknown"

                        if track_id is not None:

                            current_position = (
                                center_x,
                                center_y
                            )

                            if track_id in previous_positions:

                                old_x, old_y = (
                                    previous_positions[
                                        track_id
                                    ]
                                )

                                dy = (
                                    center_y - old_y
                                )

                                dx = (
                                    center_x - old_x
                                )

                                if abs(dy) > abs(dx):

                                    if dy > 3:
                                        direction = "Down"

                                    elif dy < -3:
                                        direction = "Up"

                                else:

                                    if dx > 3:
                                        direction = "Right"

                                    elif dx < -3:
                                        direction = "Left"

                            previous_positions[
                                track_id
                            ] = current_position

                        # -------------------------------------------------
                        # SAVE DETECTION
                        # -------------------------------------------------

                        if track_id is not None:

                            seen_objects.add(
                                (
                                    class_name,
                                    track_id
                                )
                            )

                        st.session_state.processed_data.append(
                            {
                                "frame": frame_number,
                                "id": track_id,
                                "class": class_name,
                                "confidence": round(
                                    confidence,
                                    2
                                ),
                                "x": center_x,
                                "y": center_y,
                                "direction": direction
                            }
                        )

                # -------------------------------------------------
                # WRITE OUTPUT VIDEO
                # -------------------------------------------------

                writer.write(
                    annotated_frame
                )

                # -------------------------------------------------
                # DISPLAY FRAME
                # -------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                video_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_container_width=True
                )

                # -------------------------------------------------
                # PROGRESS
                # -------------------------------------------------

                if total_frames > 0:

                    percent = (
                        frame_number /
                        total_frames
                    )

                    progress.progress(
                        min(
                            percent,
                            1.0
                        )
                    )

                status.write(
                    f"Processing frame "
                    f"{frame_number} / "
                    f"{total_frames}"
                )

                # -------------------------------------------------
                # LIVE METRICS
                # -------------------------------------------------

                unique_ids = {
                    obj[1]
                    for obj in seen_objects
                    if obj[1] is not None
                }

                cars_now = {
                    obj[1]
                    for obj in seen_objects
                    if obj[0] == "car"
                }

                motorcycles_now = {
                    obj[1]
                    for obj in seen_objects
                    if obj[0] == "motorcycle"
                }

                metric_col1.metric(
                    "Tracked Objects",
                    len(unique_ids)
                )

                metric_col2.metric(
                    "Cars",
                    len(cars_now)
                )

                metric_col3.metric(
                    "Motorcycles",
                    len(motorcycles_now)
                )

            # -------------------------------------------------
            # FINISH
            # -------------------------------------------------

            cap.release()
            writer.release()

            st.session_state.processed_video = True

            progress.progress(1.0)

            status.success(
                "✅ Traffic analysis completed."
            )

            st.success(
                "YOLO detection and object tracking "
                "completed successfully."
            )

            # -------------------------------------------------
            # DOWNLOAD PROCESSED VIDEO
            # -------------------------------------------------

            if os.path.exists(
                output_path
            ):

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_file,
                        file_name="AI_Traffic_Analysis.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

            # -------------------------------------------------
            # DOWNLOAD CSV
            # -------------------------------------------------

            if st.session_state.processed_data:

                df = pd.DataFrame(
                    st.session_state.processed_data
                )

                csv_data = df.to_csv(
                    index=False
                )

                st.download_button(
                    label="📊 Download Detection CSV",
                    data=csv_data,
                    file_name="traffic_detection_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# =========================================================
# AI TRAFFIC ASSISTANT
# =========================================================

elif page == "AI Traffic Assistant":

    st.title("🧠 AI Traffic Assistant")

    st.write(
        "Ask the AI assistant about a detected traffic event. "
        "The assistant uses the RAG knowledge base and LLM "
        "to provide an explanation."
    )

    st.divider()

    # -----------------------------------------------------
    # EVENT INPUT
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        object_type = st.selectbox(
            "Object Type",
            [
                "Car",
                "Motorcycle",
                "Bus",
                "Truck",
                "Bicycle",
                "Person"
            ]
        )

        tracking_id = st.number_input(
            "Tracking ID",
            min_value=1,
            value=17
        )

        direction = st.selectbox(
            "Direction",
            [
                "Unknown",
                "Up",
                "Down",
                "Left",
                "Right",
                "Wrong Way"
            ]
        )

    with col2:

        zone = st.selectbox(
            "Zone",
            [
                "Normal Road",
                "Restricted Zone",
                "Pedestrian Area",
                "School Zone",
                "Intersection"
            ]
        )

        speed = st.text_input(
            "Estimated Speed",
            "45 km/h"
        )

        event_type = st.selectbox(
            "Event Type",
            [
                "Normal Detection",
                "Wrong Direction",
                "Restricted Zone Entry",
                "High Traffic Density",
                "Potential Safety Concern"
            ]
        )

    st.divider()

    if st.button(
        "🧠 Analyze with RAG + LLM",
        type="primary",
        use_container_width=True
    ):

        event = {
            "object": object_type,
            "id": tracking_id,
            "direction": direction,
            "zone": zone,
            "speed": speed,
            "event": event_type
        }

        st.session_state.last_event = event

        if not LLM_AVAILABLE:

            st.warning(
                "RAG/LLM module is not currently available."
            )

            st.info(
                "Make sure rag/vector_db exists and "
                "llm/assistant.py is configured."
            )

        else:

            with st.spinner(
                "Retrieving traffic knowledge and "
                "generating AI analysis..."
            ):

                try:

                    answer = analyze_event(
                        event
                    )

                    st.success(
                        "AI analysis completed."
                    )

                    st.subheader(
                        "🤖 AI Traffic Analysis"
                    )

                    st.write(answer)

                except Exception as e:

                    st.error(
                        "Unable to generate AI analysis."
                    )

                    st.exception(e)


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("ℹ️ About the Project")

    if logo is not None:

        st.image(
            logo,
            width=220
        )

    st.header(
        "AI Smart Traffic & Road Safety Monitor"
    )

    st.write(
        """
        This project is an intelligent traffic monitoring
        system combining computer vision, object tracking,
        Retrieval-Augmented Generation (RAG), and Large
        Language Models (LLMs).
        """
    )

    st.subheader("🔍 Computer Vision")

    st.write(
        """
        YOLO is used to detect objects such as cars,
        motorcycles, buses, trucks, bicycles and people.
        """
    )

    st.subheader("🎯 Object Tracking")

    st.write(
        """
        ByteTrack assigns tracking IDs so that objects
        can be followed across video frames.
        """
    )

    st.subheader("📚 RAG")

    st.write(
        """
        The RAG component retrieves relevant traffic
        safety information from the project's knowledge base.
        """
    )

    st.subheader("🧠 LLM")

    st.write(
        """
        The LLM converts detected events and retrieved
        knowledge into a natural-language explanation.
        """
    )

    st.subheader("🚀 Main Technologies")

    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "YOLO",
                "OpenCV",
                "ByteTrack",
                "Streamlit",
                "RAG",
                "ChromaDB",
                "LLM",
                "Pandas"
            ],
            "Purpose": [
                "Application development",
                "Object detection",
                "Video processing",
                "Object tracking",
                "Web interface",
                "Knowledge retrieval",
                "Vector database",
                "AI reasoning/explanation",
                "Data analysis"
            ]
        }
    )

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.caption(
        "AI Traffic Monitor • Detect • Track • Analyze • Safer Roads"
    )