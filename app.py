import tensorflow as tf
import cv2
import numpy as np
import math
import logging
import time
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealtimeShopliftingDetector:
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
        
    def load_model(self):
        """Load the trained model with custom objects"""
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
            
            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
            logging.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            return False

    def generate_enhanced_message(self, probability, label):
        """Generate more detailed and informative messages"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if label == 0:  # Suspicious/Theft activity
            if probability <= 50:
                self.message = f"🟢 NORMAL - Regular customer behavior detected"
                severity = "LOW"
            elif probability <= 70:
                self.message = f"🟡 ALERT - Suspicious movement patterns observed"
                severity = "MEDIUM"
            elif probability <= 85:
                self.message = f"🟠 WARNING - High probability theft behavior detected"
                severity = "HIGH"
            else:
                self.message = f"🔴 CRITICAL - Very high theft risk - Immediate attention required"
                severity = "CRITICAL"
        else:  # Normal/Confusing activity
            if probability <= 50:
                self.message = f"🟢 NORMAL - Standard shopping behavior"
                severity = "LOW"
            elif probability <= 70:
                self.message = f"🟡 MONITOR - Unusual movement detected - Keep watching"
                severity = "MEDIUM"
            elif probability <= 85:
                self.message = f"🟢 LIKELY NORMAL - Behavior appears normal but monitor closely"
                severity = "LOW"
            else:
                self.message = f"🟢 NORMAL - Confirmed normal customer behavior"
                severity = "LOW"
        
        # Store detection history
        self.detection_history.append({
            'timestamp': timestamp,
            'probability': probability,
            'label': label,
            'severity': severity
        })
        
        # Keep only last 10 detections
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)

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
            logging.error(f"Frame preprocessing error: {e}")
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
            logging.error(f"Prediction error: {e}")
            return None, None

    def draw_detection_overlay(self, frame):
        """Draw detection information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main info box
        cv2.rectangle(overlay, (10, 10), (width-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Main message
        cv2.putText(frame, self.message, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence and frame info
        info_text = f"Confidence: {self.confidence}% | Frame: {self.frame_count}"
        cv2.putText(frame, info_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Processing status
        status_text = f"Buffer: {len(self.frames_queue)}/{self.sequence_length} frames"
        cv2.putText(frame, status_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Recent alerts sidebar (if any critical detections)
        recent_alerts = [d for d in self.detection_history[-3:] if d['severity'] in ['HIGH', 'CRITICAL']]
        if recent_alerts:
            alert_y = height - 100
            cv2.rectangle(frame, (10, alert_y-10), (400, height-10), (0, 0, 100), -1)
            cv2.putText(frame, "⚠️ RECENT ALERTS:", (20, alert_y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, alert in enumerate(recent_alerts):
                alert_text = f"{alert['timestamp']}: {alert['severity']} ({alert['probability']}%)"
                cv2.putText(frame, alert_text, (20, alert_y+30+i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)

    def process_video_realtime(self, video_path, display=True, save_output=None):
        """Process video in real-time with live display"""
        if not self.load_model():
            return False
            
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"❌ Cannot open video: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logging.info(f"📹 Video Info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Setup video writer if saving output
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter(save_output, fourcc, fps, frame_size)
        
        # Read first frame
        ret, previous_frame = cap.read()
        if not ret:
            logging.error("❌ Cannot read first frame")
            return False
        
        self.message = "🔄 Initializing detection system..."
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Target frame delay
        
        logging.info("🚀 Starting real-time detection...")
        start_time = time.time()
        
        try:
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    logging.info("📹 End of video reached")
                    break
                
                self.frame_count += 1
                
                # Preprocess current frame
                processed_frame = self.preprocess_frame(current_frame, previous_frame)
                if processed_frame is not None:
                    self.frames_queue.append(processed_frame)
                
                # Keep only required sequence length
                if len(self.frames_queue) > self.sequence_length:
                    self.frames_queue.pop(0)
                
                # Make prediction when we have enough frames
                if len(self.frames_queue) == self.sequence_length:
                    confidence, predicted_label = self.predict_sequence()
                    if confidence is not None:
                        self.confidence = confidence
                        self.generate_enhanced_message(confidence, predicted_label)
                        
                        # Log significant detections
                        if confidence > 70:
                            logging.warning(f"🚨 Detection: {self.message}")
                
                # Draw overlay
                display_frame = current_frame.copy()
                self.draw_detection_overlay(display_frame)
                
                # Save frame if output video specified
                if video_writer:
                    video_writer.write(display_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Real-time Shoplifting Detection', display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("🛑 User requested stop")
                        break
                    elif key == ord('p'):
                        logging.info("⏸️ Paused - Press any key to continue")
                        cv2.waitKey(0)
                    elif key == ord('s'):
                        # Save current frame
                        screenshot_path = f"detection_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_path, display_frame)
                        logging.info(f"📸 Screenshot saved: {screenshot_path}")
                
                # Update previous frame
                previous_frame = current_frame.copy()
                
                # Control playback speed
                time.sleep(frame_delay)
                
        except KeyboardInterrupt:
            logging.info("🛑 Detection stopped by user")
        except Exception as e:
            logging.error(f"❌ Error during processing: {e}")
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Final statistics
            end_time = time.time()
            processing_time = end_time - start_time
            avg_fps = self.frame_count / processing_time if processing_time > 0 else 0
            
            logging.info(f"📊 Processing completed:")
            logging.info(f"   Frames processed: {self.frame_count}")
            logging.info(f"   Processing time: {processing_time:.2f}s")
            logging.info(f"   Average FPS: {avg_fps:.2f}")
            
            # Summary of detections
            if self.detection_history:
                critical_count = sum(1 for d in self.detection_history if d['severity'] == 'CRITICAL')
                high_count = sum(1 for d in self.detection_history if d['severity'] == 'HIGH')
                logging.info(f"🚨 Critical alerts: {critical_count}")
                logging.info(f"⚠️ High alerts: {high_count}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Real-time Shoplifting Detection System')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--model', default='lrcn_160S_90_90Q.h5', help='Path to model file')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless mode)')
    parser.add_argument('--save-output', help='Path to save output video with detections')
    parser.add_argument('--frame-size', type=int, nargs=2, default=[90, 90], 
                       help='Frame size for model input (width height)')
    parser.add_argument('--sequence-length', type=int, default=160, 
                       help='Sequence length for model input')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = RealtimeShopliftingDetector(
        model_path=args.model,
        frame_width=args.frame_size[0],
        frame_height=args.frame_size[1],
        sequence_length=args.sequence_length
    )
    
    # Process video
    success = detector.process_video_realtime(
        video_path=args.video_path,
        display=not args.no_display,
        save_output=args.save_output
    )
    
    if success:
        logging.info("✅ Processing completed successfully")
    else:
        logging.error("❌ Processing failed")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
