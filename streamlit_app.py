import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import math
import time
from datetime import datetime
import tempfile
import os
import threading
import queue
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Real-time Shoplifting Detection",
    page_icon="🛡️",
    layout="wide"
)

class StreamlitShopliftingDetector:
    def __init__(self, model_path, frame_width=90, frame_height=90, sequence_length=160):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.message = ''
        self.confidence = 0
        self.frames_queue = []
        self.frame_count = 0
        self.detection_history = []
        self.is_processing = False
        self.current_frame = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the trained model with custom objects - cached for performance"""
        try:
            custom_objects = {
                'Conv2D': tf.keras.layers.Conv2D,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'TimeDistributed': tf.keras.layers.TimeDistributed,
                'LSTM': tf.keras.layers.LSTM,
                'Dense': tf.keras.layers.Dense,
                'Flatten': tf.keras.layers.Flatten,
                'Dropout': tf.keras.layers.Dropout,
                'Orthogonal': tf.keras.initializers.Orthogonal,
            }
            
            model = tf.keras.models.load_model(_self.model_path, custom_objects=custom_objects)
            return model, True, "✅ Model loaded successfully"
        except Exception as e:
            return None, False, f"❌ Failed to load model: {e}"

    def generate_enhanced_message(self, probability, label):
        """Generate more detailed and informative messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if label == 0:  # Suspicious/Theft activity
            if probability <= 50:
                self.message = f"🟢 NORMAL - Regular customer behavior detected"
                severity = "LOW"
                alert_color = "green"
            elif probability <= 70:
                self.message = f"🟡 ALERT - Suspicious movement patterns observed"
                severity = "MEDIUM"
                alert_color = "orange"
            elif probability <= 85:
                self.message = f"🟠 WARNING - High probability theft behavior detected"
                severity = "HIGH"
                alert_color = "red"
            else:
                self.message = f"🔴 CRITICAL - Very high theft risk - Immediate attention required"
                severity = "CRITICAL"
                alert_color = "red"
        else:  # Normal/Confusing activity
            if probability <= 50:
                self.message = f"🟢 NORMAL - Standard shopping behavior"
                severity = "LOW"
                alert_color = "green"
            elif probability <= 70:
                self.message = f"🟡 MONITOR - Unusual movement detected - Keep watching"
                severity = "MEDIUM"
                alert_color = "orange"
            elif probability <= 85:
                self.message = f"🟢 LIKELY NORMAL - Behavior appears normal but monitor closely"
                severity = "LOW"
                alert_color = "green"
            else:
                self.message = f"🟢 NORMAL - Confirmed normal customer behavior"
                severity = "LOW"
                alert_color = "green"
        
        # Store detection history
        detection_entry = {
            'timestamp': timestamp,
            'probability': probability,
            'label': label,
            'severity': severity,
            'message': self.message,
            'alert_color': alert_color
        }
        self.detection_history.append(detection_entry)
        
        # Keep only last 50 detections for display
        if len(self.detection_history) > 50:
            self.detection_history.pop(0)
        
        return detection_entry

    def preprocess_frame(self, current_frame, previous_frame):
        """Preprocess frames for model input"""
        try:
            # Calculate frame difference
            diff = cv2.absdiff(current_frame, previous_frame)
            diff = cv2.GaussianBlur(diff, (3, 3), 0)
            
            # Resize and convert to grayscale
            resized_frame = cv2.resize(diff, (self.frame_width, self.frame_height))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize pixel values
            normalized_frame = gray_frame / 255.0
            return normalized_frame
        except Exception as e:
            st.error(f"Frame preprocessing error: {e}")
            return None

    def predict_sequence(self):
        """Make prediction on current frame sequence"""
        try:
            if len(self.frames_queue) != self.sequence_length:
                return None, None
                
            # Prepare input for model
            input_sequence = np.expand_dims(np.array(self.frames_queue), axis=0)
            
            # Make prediction
            probabilities = self.model.predict(input_sequence, verbose=0)[0]
            predicted_label = np.argmax(probabilities)
            confidence = math.floor(max(probabilities) * 100)
            
            return confidence, predicted_label
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

def process_video_stream(detector, video_path, frame_placeholder, progress_bar, status_placeholder, detection_log):
    """Process video and update Streamlit interface"""
    # Load model
    detector.model, model_loaded, model_message = detector.load_model()
    
    if not model_loaded:
        status_placeholder.error(model_message)
        return
    
    status_placeholder.success(model_message)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_placeholder.error(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    status_placeholder.info(f"📹 Processing: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Read first frame
    ret, previous_frame = cap.read()
    if not ret:
        status_placeholder.error("❌ Cannot read first frame")
        return
    
    detector.message = "🔄 Initializing detection system..."
    frame_delay = 1.0 / fps if fps > 0 else 0.033
    
    detector.is_processing = True
    
    try:
        while detector.is_processing:
            ret, current_frame = cap.read()
            if not ret:
                status_placeholder.success("📹 Video processing completed")
                break
            
            detector.frame_count += 1
            
            # Update progress
            progress = detector.frame_count / total_frames
            progress_bar.progress(progress)
            
            # Preprocess current frame
            processed_frame = detector.preprocess_frame(current_frame, previous_frame)
            if processed_frame is not None:
                detector.frames_queue.append(processed_frame)
            
            # Keep only required sequence length
            if len(detector.frames_queue) > detector.sequence_length:
                detector.frames_queue.pop(0)
            
            # Make prediction when we have enough frames
            if len(detector.frames_queue) == detector.sequence_length:
                confidence, predicted_label = detector.predict_sequence()
                if confidence is not None:
                    detector.confidence = confidence
                    detection_entry = detector.generate_enhanced_message(confidence, predicted_label)
                    
                    # Update detection log for significant detections
                    if confidence > 60:  # Log medium and above alerts
                        with detection_log.container():
                            if detection_entry['alert_color'] == 'red':
                                st.error(f"**{detection_entry['timestamp']}** - {detection_entry['message']} (Confidence: {confidence}%)")
                            elif detection_entry['alert_color'] == 'orange':
                                st.warning(f"**{detection_entry['timestamp']}** - {detection_entry['message']} (Confidence: {confidence}%)")
                            else:
                                st.info(f"**{detection_entry['timestamp']}** - {detection_entry['message']} (Confidence: {confidence}%)")
            
            # Store current frame for display
            detector.current_frame = current_frame.copy()
            
            # Add overlay to frame
            height, width = current_frame.shape[:2]
            
            # Add detection overlay
            overlay = current_frame.copy()
            cv2.rectangle(overlay, (10, 10), (width-10, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, current_frame, 0.3, 0, current_frame)
            
            # Add text
            cv2.putText(current_frame, detector.message, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            info_text = f"Confidence: {detector.confidence}% | Frame: {detector.frame_count}"
            cv2.putText(current_frame, info_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            status_text = f"Buffer: {len(detector.frames_queue)}/{detector.sequence_length} frames"
            cv2.putText(current_frame, status_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(current_frame, timestamp, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update previous frame
            previous_frame = detector.current_frame.copy()
            
            # Control playback speed
            time.sleep(frame_delay * 0.5)  # Speed up for demo
            
    except Exception as e:
        status_placeholder.error(f"❌ Error during processing: {e}")
    finally:
        cap.release()
        detector.is_processing = False

def main():
    st.title("🛡️ Real-time Shoplifting Detection System")
    st.markdown("Upload a video to detect potential shoplifting behavior using AI")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Model path (hardcoded as requested)
        model_path = "lrcn_160S_90_90Q.h5"
        st.text_input("Model Path", value=model_path, disabled=True, help="Hardcoded model path")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            frame_width = st.number_input("Frame Width", value=90, min_value=32, max_value=224)
            frame_height = st.number_input("Frame Height", value=90, min_value=32, max_value=224)
            sequence_length = st.number_input("Sequence Length", value=160, min_value=10, max_value=300)
        
        # Detection statistics
        st.header("📊 Detection Stats")
        if 'detector' in st.session_state and st.session_state.detector.detection_history:
            detector = st.session_state.detector
            
            # Count by severity
            critical_count = sum(1 for d in detector.detection_history if d['severity'] == 'CRITICAL')
            high_count = sum(1 for d in detector.detection_history if d['severity'] == 'HIGH')
            medium_count = sum(1 for d in detector.detection_history if d['severity'] == 'MEDIUM')
            
            st.metric("🔴 Critical Alerts", critical_count)
            st.metric("🟠 High Alerts", high_count)
            st.metric("🟡 Medium Alerts", medium_count)
            st.metric("📹 Frames Processed", detector.frame_count)
        else:
            st.info("No detection data yet")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Video Feed")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for shoplifting behavior"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            # Display video info
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            st.info(f"📹 Video loaded: {width}x{height}, {fps:.2f} FPS, {duration:.2f}s duration, {total_frames} frames")
            
            # Control buttons
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("🚀 Start Detection", type="primary", disabled=st.session_state.get('processing', False)):
                    # Initialize detector
                    st.session_state.detector = StreamlitShopliftingDetector(
                        model_path=model_path,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        sequence_length=sequence_length
                    )
                    st.session_state.processing = True
                    st.session_state.temp_video_path = temp_video_path
                    st.rerun()
            
            with col_stop:
                if st.button("🛑 Stop Detection", disabled=not st.session_state.get('processing', False)):
                    if 'detector' in st.session_state:
                        st.session_state.detector.is_processing = False
                    st.session_state.processing = False
                    # Clean up temp file
                    if 'temp_video_path' in st.session_state and os.path.exists(st.session_state.temp_video_path):
                        os.unlink(st.session_state.temp_video_path)
                    st.rerun()
            
            # Video display area
            frame_placeholder = st.empty()
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Status messages
            status_placeholder = st.empty()
            
        else:
            st.info("👆 Please upload a video file to begin detection")
            frame_placeholder = st.empty()
            progress_bar = st.empty()
            status_placeholder = st.empty()
    
    with col2:
        st.header("🚨 Real-time Detection Log")
        detection_log = st.container()
        
        # Display current detection status
        if 'detector' in st.session_state and st.session_state.detector.message:
            detector = st.session_state.detector
            st.subheader("Current Status")
            
            # Color code based on severity
            if "CRITICAL" in detector.message:
                st.error(f"**{detector.message}**")
            elif "WARNING" in detector.message or "ALERT" in detector.message:
                st.warning(f"**{detector.message}**")
            else:
                st.success(f"**{detector.message}**")
            
            st.metric("Confidence", f"{detector.confidence}%")
        
        # Live detection log
        with detection_log:
            if 'detector' in st.session_state and st.session_state.detector.detection_history:
                st.subheader("Recent Detections")
                
                # Show recent detections (last 10)
                recent_detections = st.session_state.detector.detection_history[-10:]
                
                for detection in reversed(recent_detections):  # Show newest first
                    if detection['severity'] in ['CRITICAL', 'HIGH']:
                        if detection['alert_color'] == 'red':
                            st.error(f"**{detection['timestamp']}** - {detection['severity']} - {detection['probability']}%")
                        else:
                            st.warning(f"**{detection['timestamp']}** - {detection['severity']} - {detection['probability']}%")
                    elif detection['severity'] == 'MEDIUM':
                        st.warning(f"**{detection['timestamp']}** - {detection['severity']} - {detection['probability']}%")
                    else:
                        st.info(f"**{detection['timestamp']}** - {detection['severity']} - {detection['probability']}%")
            else:
                st.info("No detections yet - start processing to see real-time alerts")
    
    # Process video if started
    if st.session_state.get('processing', False) and 'detector' in st.session_state:
        detector = st.session_state.detector
        video_path = st.session_state.temp_video_path
        
        # Run processing in the main thread for Streamlit
        if not detector.is_processing:
            detector.is_processing = True
            
            # Start processing
            process_video_stream(detector, video_path, frame_placeholder, progress_bar, status_placeholder, detection_log)
            
            # Auto-stop when done
            st.session_state.processing = False
            detector.is_processing = False
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions:
    1. **Upload** a video file using the file uploader
    2. **Click** 'Start Detection' to begin real-time analysis
    3. **Monitor** the live video feed and detection log
    4. **Click** 'Stop Detection' to end processing
    
    ### 🎯 Detection Levels:
    - 🟢 **NORMAL**: Regular customer behavior
    - 🟡 **MEDIUM**: Suspicious movement patterns
    - 🟠 **HIGH**: High probability theft behavior
    - 🔴 **CRITICAL**: Very high theft risk - immediate attention required
    """)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False

if __name__ == "__main__":
    main()