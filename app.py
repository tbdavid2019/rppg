#!/usr/bin/env python3
"""
rPPG Heart Rate Estimation using OpenCV and POS algorithm
"""
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import tempfile
import time
from tqdm import tqdm

class SimpleRPPG:
    def __init__(self, min_bpm=45, max_bpm=180):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame):
        """Detect faces using OpenCV Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Try multiple parameter sets for better detection
        param_sets = [
            {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (50, 50)},
            {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (30, 30)},
            {"scaleFactor": 1.2, "minNeighbors": 6, "minSize": (60, 60)},
        ]
        
        for params in param_sets:
            faces = self.face_cascade.detectMultiScale(gray, **params)
            if len(faces) > 0:
                return faces
        
        return []
    
    def extract_roi_signal(self, frame, face_box):
        """Extract ROI and compute mean RGB values"""
        x, y, w, h = face_box
        
        # Define ROI (forehead and cheek areas)
        roi_y1 = y + int(0.2 * h)
        roi_y2 = y + int(0.7 * h)
        roi_x1 = x + int(0.15 * w)
        roi_x2 = x + int(0.85 * w)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return None
        
        # Calculate mean RGB values
        mean_rgb = np.mean(roi, axis=(0, 1))
        return mean_rgb
    
    def pos_algorithm(self, rgb_signals, fps):
        """POS (Plane-Orthogonal-to-Skin) algorithm"""
        if len(rgb_signals) < 30:  # Need at least 1 second of data at 30fps
            return None, None
        
        rgb_signals = np.array(rgb_signals)
        
        # Normalize RGB signals
        mean_rgb = np.mean(rgb_signals, axis=0)
        normalized_rgb = rgb_signals / mean_rgb
        
        # POS algorithm
        X1 = normalized_rgb[:, 0] - normalized_rgb[:, 1]  # R - G
        X2 = normalized_rgb[:, 0] + normalized_rgb[:, 1] - 2 * normalized_rgb[:, 2]  # R + G - 2B
        
        # Temporal filtering (bandpass)
        low_freq = self.min_bpm / 60.0
        high_freq = self.max_bpm / 60.0
        
        sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=fps, output='sos')
        X1_filtered = signal.sosfilt(sos, X1)
        X2_filtered = signal.sosfilt(sos, X2)
        
        # POS combination
        alpha = np.std(X1_filtered) / np.std(X2_filtered)
        pulse_signal = X1_filtered - alpha * X2_filtered
        
        return pulse_signal, self.estimate_heart_rate(pulse_signal, fps)
    
    def estimate_heart_rate(self, pulse_signal, fps):
        """Estimate heart rate using FFT"""
        if len(pulse_signal) < fps:  # Need at least 1 second
            return None
        
        # Apply window function
        windowed_signal = pulse_signal * signal.windows.hann(len(pulse_signal))
        
        # FFT
        freqs = fftfreq(len(windowed_signal), 1/fps)
        fft_values = np.abs(fft(windowed_signal))
        
        # Find frequency range corresponding to heart rate
        min_freq = self.min_bpm / 60.0
        max_freq = self.max_bpm / 60.0
        
        valid_indices = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(valid_indices):
            return None
        
        valid_freqs = freqs[valid_indices]
        valid_fft = fft_values[valid_indices]
        
        # Find peak frequency
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        
        heart_rate = peak_freq * 60.0
        
        # Confidence based on peak prominence
        confidence = np.max(valid_fft) / np.mean(valid_fft)
        confidence = min(confidence / 10.0, 1.0)  # Normalize to 0-1
        
        return {"hr": heart_rate, "confidence": confidence}
    
    def process_video(self, video_path, window_seconds=10.0, step_seconds=2.0, conf_threshold=0.3, progress_callback=None):
        """Process video and extract heart rate"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            return [], [], []
        
        window_frames = int(window_seconds * fps)
        step_frames = int(step_seconds * fps)
        
        results_time = []
        results_hr = []
        results_conf = []
        
        frame_buffer = []
        rgb_buffer = []
        
        frame_idx = 0
        processed_chunks = 0
        
        # Console progress bar
        pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
        
        # First check for face detection
        if progress_callback:
            progress_callback(0.1, "🔍 檢測人臉中...")
        
        face_found = False
        for i in range(0, min(300, total_frames), 30):  # Check first 10 seconds
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detect_faces(rgb_frame)
                if len(faces) > 0:
                    face_found = True
                    if progress_callback:
                        progress_callback(0.15, f"✅ 在第 {i} 幀 ({i/fps:.1f}秒) 檢測到人臉！")
                    break
        
        if not face_found:
            if progress_callback:
                progress_callback(0.15, "⚠️ 未檢測到人臉，繼續處理...")
        
        # Reset to beginning and process in chunks
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        estimated_chunks = max(1, (total_frames - window_frames) // step_frames + 1)
        pbar.reset(total=estimated_chunks)
        pbar.set_description("Processing chunks")
        
        processed_chunks = 0
        
        # Process video in chunks (much more efficient)
        for chunk_start in range(0, total_frames - window_frames + 1, step_frames):
            chunk_frames = []
            
            # Read a batch of frames for this chunk
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
            batch_frames = []
            
            # Read all frames for this window at once
            for i in range(window_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_frames.append(rgb_frame)
            
            # Detect face only in the first frame of the batch
            if len(batch_frames) > 0:
                faces = self.detect_faces(batch_frames[0])
                if len(faces) > 0:
                    current_face_box = max(faces, key=lambda x: x[2] * x[3])
                    
                    # Extract signals from all frames using the same face box
                    for rgb_frame in batch_frames:
                        rgb_signal = self.extract_roi_signal(rgb_frame, current_face_box)
                        if rgb_signal is not None:
                            chunk_frames.append(rgb_signal)
            
            # Process this chunk if we have enough data
            if len(chunk_frames) >= fps:  # Need at least 1 second of data
                pulse_signal, hr_result = self.pos_algorithm(chunk_frames, fps)
                
                if hr_result is not None and hr_result["hr"] > 0 and hr_result["confidence"] >= conf_threshold:
                    t_sec = (chunk_start + window_frames // 2) / fps  # Center time of window
                    results_time.append(t_sec)
                    results_hr.append(hr_result["hr"])
                    results_conf.append(hr_result["confidence"])
                    
                    print(f"✅ Chunk {processed_chunks + 1}: HR = {hr_result['hr']:.1f} BPM at {t_sec:.1f}s")
            
            processed_chunks += 1
            pbar.update(1)
            
            # Update Gradio progress
            if progress_callback:
                progress_val = 0.15 + (processed_chunks / estimated_chunks) * 0.7
                if len(results_hr) > 0:
                    progress_callback(progress_val, f"💓 找到 {len(results_hr)} 個心率測量值")
                else:
                    progress_callback(progress_val, f"處理第 {processed_chunks}/{estimated_chunks} 段...")
            
            # Early termination if we have enough successful measurements
            if len(results_hr) >= 10:  # Stop if we have 10 good measurements
                print(f"✅ Early termination: Found {len(results_hr)} measurements")
                break
        
        cap.release()
        pbar.close()  # Close console progress bar
        
        if progress_callback:
            progress_callback(1.0, f"完成！找到 {len(results_hr)} 個心率測量值")
        
        return results_time, results_hr, results_conf

def quick_face_check(video_path, progress=None):
    """Quick face detection check"""
    if not video_path:
        return "請先上傳影片檔案"
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if progress:
        progress(0.1, "🎬 開始檢查影片...")
    
    # 載入 OpenCV 人臉檢測器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Console progress bar for face detection
    face_pbar = tqdm(total=total_frames//15, desc="Face detection", unit="frames")
    
    face_detected = False
    face_found_at_frame = None
    
    for i in range(0, total_frames, 15):  # 每隔15幀檢查一次
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 嘗試多種參數組合
            param_sets = [
                {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)},
                {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (20, 20)},
                {"scaleFactor": 1.2, "minNeighbors": 6, "minSize": (40, 40)},
            ]
            
            faces_found = False
            for params in param_sets:
                faces = face_cascade.detectMultiScale(gray, **params)
                if len(faces) > 0:
                    faces_found = True
                    face_detected = True
                    face_found_at_frame = i
                    time_stamp = i / fps
                    
                    if progress:
                        progress(0.8, f"✅ 在第 {i} 幀 ({time_stamp:.1f}秒) 檢測到 {len(faces)} 個人臉！")
                    break
            
            if faces_found:
                break
        
        face_pbar.update(1)  # Update console progress bar
        
        # 更新檢測進度
        if progress and i % 150 == 0:
            detection_progress = 0.1 + min((i / total_frames) * 0.7, 0.7)
            current_time = i / fps
            progress(detection_progress, f"🔍 檢測人臉中... 已檢查到 {current_time:.1f}秒")
    
    cap.release()
    face_pbar.close()  # Close console progress bar
    
    if face_detected:
        success_msg = f"✅ 成功！在第 {face_found_at_frame} 幀 ({face_found_at_frame/fps:.1f}秒) 檢測到人臉"
        if progress:
            progress(1.0, success_msg)
        return success_msg + "\n\n💡 這個影片適合進行心率分析！"
    else:
        fail_msg = "❌ 整個影片中未檢測到人臉"
        if progress:
            progress(1.0, fail_msg)
        return fail_msg + "\n\n📋 建議:\n• 確保影片中有清晰的正面人臉\n• 檢查光線是否充足\n• 避免過度的頭部移動"

def process_video(video_path, method, window, step, min_bpm, max_bpm, conf, progress=gr.Progress()):
    """Process video and extract heart rate"""
    if not video_path:
        return "請上傳影片檔案", None, None
    
    start_time = time.time()
    print(f"🚀 開始處理影片: {video_path}")
    
    # Initialize rPPG processor
    rppg = SimpleRPPG(min_bpm=min_bpm, max_bpm=max_bpm)
    
    # Process video
    ts, hr, cf = rppg.process_video(
        video_path, 
        window_seconds=window, 
        step_seconds=step, 
        conf_threshold=conf,
        progress_callback=progress
    )
    
    processing_time = time.time() - start_time
    print(f"⏱️  處理完成！耗時: {processing_time:.1f} 秒，找到 {len(hr)} 個心率測量值")
    
    if not hr:
        return f"未檢測到心率數據。處理時間: {processing_time:.1f}秒", None, None
    
    # Create CSV
    csv_content = "time_sec,hr_bpm,confidence\n"
    for a, b, c in zip(ts, hr, cf):
        csv_content += f"{a:.2f},{b:.2f},{c:.3f}\n"
    
    # Create plot
    plt.figure(figsize=(10, 4))
    plt.plot(ts, hr, 'b-', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title(f"Heart Rate Estimation (Avg: {np.mean(hr):.1f} BPM)")
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        plot_path = tmp.name
    
    plt.close()
    
    # Save CSV to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.write(csv_content)
        csv_path = tmp.name
    
    result_msg = f"✅ 成功分析！\n平均心率: {np.mean(hr):.1f} BPM\n測量點數: {len(hr)}\n處理時間: {processing_time:.1f} 秒"
    
    return result_msg, plot_path, csv_path

# Gradio interface
with gr.Blocks(title="rPPG Heart Rate Analysis") as demo:
    gr.Markdown("# rPPG Heart Rate Analysis")
    gr.Markdown("Upload a video to estimate heart rate using computer vision.")
    
    with gr.Tabs():
        with gr.Tab("Heart Rate Analysis"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    
                    with gr.Row():
                        method_select = gr.Dropdown(
                            choices=["POS"],
                            value="POS",
                            label="Method"
                        )
                        
                        conf_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Confidence Threshold"
                        )
                    
                    with gr.Row():
                        window_slider = gr.Slider(
                            minimum=5.0,
                            maximum=30.0,
                            value=10.0,
                            step=1.0,
                            label="Window (sec)"
                        )
                        
                        step_slider = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=2.0,
                            step=0.5,
                            label="Step (sec)"
                        )
                    
                    with gr.Row():
                        min_bpm = gr.Slider(
                            minimum=30,
                            maximum=100,
                            value=45,
                            step=5,
                            label="Min BPM"
                        )
                        
                        max_bpm = gr.Slider(
                            minimum=100,
                            maximum=200,
                            value=180,
                            step=5,
                            label="Max BPM"
                        )
                    
                    process_btn = gr.Button("Process Video", variant="primary", size="lg")
                
                with gr.Column():
                    result_text = gr.Textbox(label="Results", lines=4)
                    plot_output = gr.Image(label="Heart Rate Plot")
                    csv_output = gr.File(label="Download CSV Data")
        
        with gr.Tab("Face Detection Test"):
            with gr.Row():
                with gr.Column():
                    test_video_input = gr.Video(label="Upload Video for Face Test")
                    check_btn = gr.Button("Test Face Detection", variant="secondary", size="lg")
                
                with gr.Column():
                    check_result = gr.Textbox(label="Face Detection Results", lines=8)
    
    # Connect functions
    process_btn.click(
        fn=process_video,
        inputs=[video_input, method_select, window_slider, step_slider, min_bpm, max_bpm, conf_slider],
        outputs=[result_text, plot_output, csv_output],
        show_progress=True
    )
    
    check_btn.click(
        fn=quick_face_check,
        inputs=[test_video_input],
        outputs=[check_result],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )