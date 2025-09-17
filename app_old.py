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

def process_video(video_path, method, window, step, min_bpm, max_bpm, conf, progress=None):
    import time
    from tqdm import tqdm
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file.", None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    if not fps or fps <= 1e-3:
        fps = 30.0
    
    win_frames = int(window * fps)
    step_frames = int(step * fps)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.1f}s duration")
    print(f"Processing parameters: window={window}s ({win_frames} frames), step={step}s ({step_frames} frames)")
    
    # Estimate processing time
    estimated_chunks = max(1, (total_frames - win_frames) // step_frames + 1)
    print(f"Estimated {estimated_chunks} processing chunks")
    
    # Reliable face detection using OpenCV
    print("🔍 Using reliable OpenCV face detection...")
    if progress:
        progress(0.05, "🔍 使用 OpenCV 檢測人臉中...")
    
    # 載入 OpenCV 人臉檢測器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_detected = False
    face_found_at_frame = None
    checked_frames = 0
    
    # 持續檢測直到找到人臉為止（每隔15幀檢查一次以提高準確性）
    for i in range(0, total_frames, 15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # 使用 OpenCV 進行人臉檢測
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 多種參數組合來提高檢測率
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
                        
                        # 立即顯示 Gradio 提示
                        if progress:
                            progress(0.1, f"✅ 在第 {i} 幀 ({time_stamp:.1f}秒) 檢測到 {len(faces)} 個人臉！開始處理...")
                        print(f"✅ Face detected at frame {i} ({time_stamp:.1f}s)! Found {len(faces)} faces")
                        break
                
                if faces_found:
                    break
                    
            except Exception as e:
                print(f"OpenCV 檢測錯誤: {e}")
                pass
        
        checked_frames += 1
        # 更新檢測進度
        if progress and checked_frames % 20 == 0:  # 每檢查20幀更新一次進度
            detection_progress = 0.05 + min((i / total_frames) * 0.05, 0.05)
            current_time = i / fps
            progress(detection_progress, f"🔍 檢測人臉中... 已檢查到 {current_time:.1f}秒 ({checked_frames} 次檢查)")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    if not face_detected:
        warning_msg = "⚠️ 警告：整個影片中未檢測到人臉！"
        print("⚠️  WARNING: No faces detected in entire video!")
        print("💡 This video may not contain detectable faces. Processing will continue but will likely fail.")
        if progress:
            progress(0.1, warning_msg)
    else:
        success_msg = f"✅ 人臉檢測成功 - 在第 {face_found_at_frame} 幀 ({face_found_at_frame/fps:.1f}秒) 找到人臉"
        print("✅ Face detection working - processing should succeed")
        if progress:
            progress(0.15, success_msg)

    # Map string methods to Method enum
    method_map = {
        "pos": Method.POS,
        "chrom": Method.CHROM,
        "g": Method.G,
        "vitallens": Method.VITALLENS
    }
    
    rppg_method = method_map.get(method.lower(), Method.POS)
    
    # Create VitalLens instance with optimized parameters
    rppg = VitalLens(
        method=rppg_method,
        detect_faces=True,           # 啟用人臉檢測
        estimate_rolling_vitals=False,  # 停用滾動估計以提高穩定性
        roi_method="MEDIAN",         # 使用中位數 ROI 方法
        mode="BATCH"                 # 使用批處理模式
    )

    # Process video in chunks instead of loading all frames
    ts, hr, cf = [], [], []
    frame_buffer = []
    frame_idx = 0
    processed_chunks = 0
    
    # Progress bar for console
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    
    start_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb)
        frame_idx += 1
        
        # Update progress
        pbar.update(1)
        if progress:
            progress(frame_idx / total_frames, f"Reading frames: {frame_idx}/{total_frames}")
        
        # Process when we have enough frames for a window and it's time to process
        if len(frame_buffer) >= win_frames and (frame_idx % step_frames == 0 or frame_idx == total_frames):
            try:
                # Use the last window_frames for processing
                window_frames = frame_buffer[-win_frames:] if len(frame_buffer) >= win_frames else frame_buffer
                video_chunk = np.array(window_frames)
                
                # Process this chunk
                chunk_results = rppg(video_chunk, fps=fps)
                
                # Extract results
                if chunk_results and len(chunk_results) > 0:
                    result = chunk_results[-1]  # Take the last (most recent) result
                    
                    # Try different possible key names
                    hr_value = None
                    conf_value = 0.0
                    
                    if isinstance(result, dict):
                        # Try common key names for heart rate
                        for hr_key in ['hr', 'heart_rate', 'bpm', 'pulse']:
                            if hr_key in result:
                                hr_value = result[hr_key]
                                break
                        
                        # Try common key names for confidence
                        for conf_key in ['confidence', 'conf', 'reliability', 'quality']:
                            if conf_key in result:
                                conf_value = result[conf_key]
                                break
                    
                    if hr_value is not None and conf_value >= conf:
                        t_sec = frame_idx / fps
                        ts.append(t_sec)
                        hr.append(float(hr_value))
                        cf.append(float(conf_value))
                        print(f"✅ Found HR: t={t_sec:.1f}s, HR={hr_value:.1f}, conf={conf_value:.3f}")
                        
                        # Gradio 更新：顯示找到的心率
                        if progress:
                            progress_val = min(processed_chunks / estimated_chunks, 0.95)
                            progress(progress_val, f"💓 找到心率: {hr_value:.1f} BPM (第 {processed_chunks} 段)")
                            
                    elif processed_chunks % 50 == 0:  # Print status every 50 chunks
                        face_status = "有人臉" if len(chunk_results) > 0 else "無人臉"
                        print(f"⚠️  No valid HR found in chunk {processed_chunks}. Face detected: {len(chunk_results) > 0}")
                        
                        # Gradio 更新：顯示處理狀態
                        if progress:
                            progress_val = min(processed_chunks / estimated_chunks, 0.95)
                            progress(progress_val, f"⚠️ 處理中... 第 {processed_chunks} 段 ({face_status})")
                            
                else:
                    if processed_chunks % 50 == 0:  # Print status every 50 chunks
                        print(f"❌ No results from chunk {processed_chunks}. Possible face detection failure.")
                        
                        # Gradio 更新：顯示檢測失敗
                        if progress:
                            progress_val = min(processed_chunks / estimated_chunks, 0.95)
                            progress(progress_val, f"❌ 第 {processed_chunks} 段：人臉檢測可能失敗")
                
                # 早期停止機制：如果前面的 chunks 都沒有找到有效測量值
                if processed_chunks >= 10 and len(hr) == 0:
                    stop_msg = f"🛑 提前停止：前 {processed_chunks+1} 段都未找到有效測量值"
                    print(f"\n🛑 EARLY STOP: No valid measurements found in first {processed_chunks+1} chunks.")
                    print("💡 This indicates the video is not suitable for rPPG analysis.")
                    print("📋 Suggestions:")
                    print("   - Ensure clear, front-facing face visibility")
                    print("   - Good lighting conditions") 
                    print("   - Minimal head movement")
                    print("   - Try a different video")
                    
                    # Gradio 更新：顯示停止原因
                    if progress:
                        progress(0.5, f"{stop_msg} - 影片可能不適合分析")
                    break
                        
                processed_chunks += 1
                elapsed = time.time() - start_time
                if processed_chunks > 0:
                    avg_time_per_chunk = elapsed / processed_chunks
                    remaining_chunks = estimated_chunks - processed_chunks
                    eta = remaining_chunks * avg_time_per_chunk
                    print(f"Processed chunk {processed_chunks}/{estimated_chunks}, ETA: {eta:.1f}s")
                    
                    # 定期更新 Gradio 進度（每10個chunk或每20秒更新一次）
                    if processed_chunks % 10 == 0 or (processed_chunks > 0 and processed_chunks % max(1, int(20 / avg_time_per_chunk)) == 0):
                        progress_val = min(processed_chunks / estimated_chunks, 0.95)
                        hr_count = len(hr)
                        if progress:
                            if hr_count > 0:
                                latest_hr = hr[-1] if hr else 0
                                progress(progress_val, f"💓 已找到 {hr_count} 個心率測量值 (最新: {latest_hr:.1f} BPM) - 進度: {processed_chunks}/{estimated_chunks}")
                            else:
                                progress(progress_val, f"🔄 處理中... {processed_chunks}/{estimated_chunks} 段 (預估剩餘: {eta:.0f}秒)")
                
            except Exception as e:
                print(f"Error processing chunk at frame {frame_idx}: {e}")
                continue
            
            # Keep only recent frames to manage memory
            if len(frame_buffer) > win_frames * 2:
                frame_buffer = frame_buffer[-win_frames:]
    
    pbar.close()
    cap.release()
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.1f}s, extracted {len(hr)} heart rate measurements")
    
    # Analysis and suggestions
    if len(hr) == 0:
        print("\n⚠️  WARNING: No heart rate measurements extracted!")
        print("💡 Possible causes:")
        print("   1. No face detected in the video")
        print("   2. Poor lighting conditions")
        print("   3. Face too small or partially occluded")
        print("   4. Excessive head movement")
        print("   5. Video quality too low")
        print("   6. Confidence threshold too high")
        print("\n🔧 Suggestions:")
        print("   - Ensure the face is clearly visible and well-lit")
        print("   - Try lowering the confidence threshold to 0.0")
        print("   - Use a shorter video segment for testing")
        print("   - Ensure minimal head movement")
    elif len(hr) < 5:
        print(f"\n⚠️  WARNING: Only {len(hr)} measurements extracted from {processed_chunks} chunks")
        print("💡 This might indicate intermittent face detection issues")
    else:
        print(f"\n✅ Successfully extracted {len(hr)} measurements from {processed_chunks} chunks")
        print(f"📊 Success rate: {len(hr)/processed_chunks*100:.1f}%")

    if not hr:
        return "No heart rate data extracted.", None, None

    # Create CSV
    csv_content = "time_sec,hr_bpm,confidence\n"
    for a, b, c in zip(ts, hr, cf):
        csv_content += f"{a:.2f},{b:.2f},{c:.3f}\n"

    # Create plot
    plt.figure(figsize=(10,4))
    plt.plot(ts, hr)
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Estimated HR from rPPG (video)")
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        plot_path = tmp.name
    
    plt.close()

    return f"Processed {len(hr)} data points.", plot_path, csv_content

def gradio_interface(video, method, window, step, min_bpm, max_bpm, conf, progress=gr.Progress()):
    if video is None:
        return "Please upload a video file.", None, None
    
    # video is a file object from Gradio, get the file path as string
    video_path = str(video.name) if hasattr(video, 'name') else str(video)
    
    # Ensure it's an absolute path
    if not os.path.isabs(video_path):
        video_path = os.path.abspath(video_path)
    
    try:
        # Pass progress callback to process_video
        def progress_callback(pct, msg):
            progress(pct, desc=msg)
        
        message, plot_path, csv_content = process_video(video_path, method, window, step, min_bpm, max_bpm, conf, progress_callback)
        if plot_path:
            return message, plot_path, csv_content
        else:
            return message, None, None
    except Exception as e:
        return f"Error processing video: {str(e)}", None, None

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# rPPG Heart Rate Estimation from Video")
    gr.Markdown("Upload a video file to estimate heart rate using rPPG methods.")
    
    with gr.Row():
        video_input = gr.File(label="Upload Video", file_types=[".mp4", ".mkv", ".mov", ".avi"])
    
    with gr.Row():
        method = gr.Dropdown(["pos", "chrom", "g"], value="pos", label="Method")
        window = gr.Number(value=10.0, label="Window (seconds)")
        step = gr.Number(value=2.0, label="Step (seconds)")
    
    with gr.Row():
        min_bpm = gr.Number(value=42, label="Min BPM")
        max_bpm = gr.Number(value=180, label="Max BPM")
        conf = gr.Number(value=0.0, label="Confidence Threshold (0.0 = accept all)")
    
    submit_btn = gr.Button("Process Video")
    
    output_text = gr.Textbox(label="Status")
    output_plot = gr.Image(label="Heart Rate Plot")
    output_csv = gr.Textbox(label="CSV Data", lines=10)
    
    submit_btn.click(
        gradio_interface,
        inputs=[video_input, method, window, step, min_bpm, max_bpm, conf],
        outputs=[output_text, output_plot, output_csv]
    )

if __name__ == "__main__":
    demo.launch()