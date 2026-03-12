import os
import asyncio
import logging
import tempfile
from typing import Dict, Any, Tuple

from app.services.video.video_frame_extractor import extract_video_frames
from app.services.video.video_ocr_service import extract_text_from_frame
from app.services.video.video_text_aggregator import aggregate_ocr_text
from app.services.video.video_ai_detector import analyze_video_ai
from app.services.video.video_frame_hasher import compute_video_hashes
from app.services.video.video_hash_database import get_all_hashes, store_frame_hash
from app.services.video.video_context_detector import detect_video_reuse
from app.services.video.video_scene_describer import generate_video_scene_description
from app.services.video_analysis import extract_transcript

async def process_video_text(video_url: str) -> Dict[str, Any]:
    """
    Complete Phase A Multi-modal pipeline:
      1. Synchronously extracts audio and transcribes.
      2. Synchronously extracts video frames using ffmpeg.
      3. Asynchronously runs Azure OCR, AI Detection, and Visual Scene Description.
      4. Aggregates results into a unified payload for the LLM.
    """
    print(f"🎬 [Pipeline] Starting Multi-modal Video Pipeline for: {video_url}")
    
    # 1. Run the existing Audio Extractor
    print("🎬 [Pipeline] Step 1: Extracting Audio Transcript")
    audio_result = await asyncio.to_thread(extract_transcript, video_url)
    spoken_transcript = audio_result.get("transcript") if audio_result.get("success") else None
    
    audio_present = False
    if spoken_transcript and len(str(spoken_transcript).strip()) > 0:
        audio_present = True
    
    print(f"🎙️ [Pipeline] Video transcript detected: {audio_present}")
    
    # 2. Extract Video Frames
    print("🎬 [Pipeline] Step 2: Extracting Video Frames via FFmpeg")
    frame_paths = []
    
    # Use centralized TEMP_DIR from Config
    from app.config.settings import Config
    with tempfile.TemporaryDirectory(prefix="trustlens_vid_frames_", dir=Config.TEMP_DIR) as tmp_dir:
        try:
            # Upgrade per requirements: 1 FPS, max 20 frames
            frame_paths = await asyncio.to_thread(
                extract_video_frames,
                video_url, 
                tmp_dir, 
                fps=1.0, 
                max_frames=20
            )
            print(f"🖼️ [Pipeline] Frames extracted: {len(frame_paths)}")
        except Exception as e:
            print(f"❌ [Pipeline] Frame Extraction failed: {e}")
            
        # 2.5 Compute frame hashes for context reuse detection
        frame_hashes = []
        context_detection_result = {
            "contextReuseDetected": False,
            "matchedFrames": 0,
            "confidence": 0.0,
            "matchedSources": []
        }
        
        if frame_paths:
            try:
                # Use frame_paths instead of the incorrect frame_hashes variable from previous version
                frame_hashes = await asyncio.to_thread(compute_video_hashes, frame_paths)
                if frame_hashes:
                    database_hashes = await asyncio.to_thread(get_all_hashes)
                    context_detection_result = await asyncio.to_thread(
                        detect_video_reuse, frame_hashes, database_hashes
                    )
            except Exception as e:
                print(f"❌ [Pipeline] Context Reuse Detection failed: {e}")

        # 3. Process Frames in Parallel (OCR + AI Detection + Visual Scene Description)
        print(f"🎬 [Pipeline] Step 3: Running Async Analysis on {len(frame_paths)} frames")
        all_frame_texts = []
        ai_detection_result = {
            "aiGeneratedProbability": 0.0,
            "isLikelyAIGenerated": False,
            "framesAnalyzed": 0
        }
        visual_summary = "No visual events detected."
        events_detected = []
        
        if frame_paths:
            # Optimize OCR by sampling the same 8 frames as the scene describer
            # This significantly reduces Azure Vision costs and latency
            num_frames = len(frame_paths)
            if num_frames > 8:
                indices = [int(i * (num_frames - 1) / 7) for i in range(8)]
                ocr_frame_paths = [frame_paths[i] for i in indices]
            else:
                ocr_frame_paths = frame_paths
                
            async def run_ocr():
                print(f"🎬 [Pipeline] OCR Analysis on {len(ocr_frame_paths)} sample frames...")
                ocr_tasks = [extract_text_from_frame(fp) for fp in ocr_frame_paths]
                res = await asyncio.gather(*ocr_tasks, return_exceptions=True)
                return [r if not isinstance(r, Exception) else [] for r in res]
                
            ocr_coro = run_ocr()
            ai_coro = analyze_video_ai(frame_paths)
            visual_coro = generate_video_scene_description(frame_paths)
            
            results = await asyncio.gather(ocr_coro, ai_coro, visual_coro, return_exceptions=True)
            
            all_frame_texts = results[0] if not isinstance(results[0], Exception) else []
            if not isinstance(results[1], Exception):
                ai_detection_result = results[1]
            
            if not isinstance(results[2], Exception):
                visual_summary = results[2].get("scene_summary", visual_summary)
                events_detected = results[2].get("events_detected", [])
                print("🎬 [Pipeline] Visual scene summary generated")
            
        # 4. Synthesize & Structure
        print("🎬 [Pipeline] Step 4: Aggregating Multi-modal Outputs")
        capped_lines, ocr_metadata = aggregate_ocr_text(all_frame_texts)
        
        combined_text = ""
        if audio_present:
            combined_text += f"AUDIO TRANSCRIPT:\n[{spoken_transcript}]\n\n"
        else:
            combined_text += "[Note: Video has no audible speech or transcript]\n\n"
            
        if visual_summary:
            combined_text += f"VISUAL SCENE ANALYSIS:\n{visual_summary}\n\n"
            
        if capped_lines:
            combined_text += "ON-SCREEN TEXT DETECTED:\n"
            for line in capped_lines:
                combined_text += f"[{line}]\n"
                
        if not combined_text or combined_text.strip() == "[Note: Video has no audible speech or transcript]":
            combined_text = "[No audible transcript, visual evidence, or visible text detected in this video]"

        visual_text_detected = len(capped_lines) > 0
        
        # 5. Store new video hashes
        if frame_hashes:
            print("🎬 [Pipeline] Step 5: Archiving video hashes")
            hash_metadata = {
                "video_id": audio_result.get("video_id"),
                "platform": audio_result.get("platform", "unknown"),
                "source_url": video_url
            }
            try:
                for h in frame_hashes:
                    await asyncio.to_thread(store_frame_hash, h, hash_metadata)
            except Exception as e:
                print(f"⚠️ [Pipeline] Hash storage failed: {e}")

        return {
            "success": True,
            "videoTranscript": spoken_transcript,
            "audioPresent": audio_present,
            "visualSummary": visual_summary,
            "eventsDetected": events_detected,
            "ocrText": capped_lines,
            "combinedVideoText": combined_text,
            "framesAnalyzed": len(frame_paths),
            "visualTextDetected": visual_text_detected,
            "ocrMetadata": ocr_metadata,
            "method": "multimodal_pipeline",
            "platform": audio_result.get("platform", "unknown"),
            "video_id": audio_result.get("video_id"),
            "title": audio_result.get("title", ""),
            "aiDetection": ai_detection_result,
            "contextDetection": context_detection_result,
            "error": audio_result.get("error") if not audio_result.get("success") else None
        }
