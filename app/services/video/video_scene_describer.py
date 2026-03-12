import json
import base64
import asyncio
from typing import List, Dict, Any
from app.config.azure import get_azure_client
from app.config.settings import Config
from app.services.llm_analysis import detect_image_mime_type

async def generate_video_scene_description(frame_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze extracted video frames using Azure OpenAI Vision to describe factual events.
    Returns a scene summary and a list of detected events.
    """
    if not frame_paths:
        return {"scene_summary": "No frames provided for visual analysis.", "events_detected": []}

    try:
        client = get_azure_client()
        
        # Optimize payload by sampling 8 chronological frames instead of 20
        # This reduces processing time and prevents timeouts while maintaining context
        num_frames = len(frame_paths)
        if num_frames > 8:
            indices = [int(i * (num_frames - 1) / 7) for i in range(8)]
            frames_to_send = [frame_paths[i] for i in indices]
        else:
            frames_to_send = frame_paths
        
        print(f"🎬 [Scene Describer] Analyzing {len(frames_to_send)} sample frames...")
        
        image_contents = []
        for path in frames_to_send:
            with open(path, "rb") as f:
                image_bytes = f.read()
                mime_type = detect_image_mime_type(image_bytes)
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })

        system_prompt = (
            "You are a visual scene analyst specializing in event detection. "
            "The following images are chronological frames extracted from a single video. "
            "Describe the factual events occurring in the video accurately and objectively. "
            "Focus on: explosions, aircraft movements/crashes, protests, military actions, "
            "natural disasters, and public gatherings.\n\n"
            "Return ONLY a JSON object with two keys:\n"
            "1. 'scene_summary': A concise paragraph describing the main visual evidence.\n"
            "2. 'events_detected': A list of specific factual labels (e.g., ['missile strike', 'building fire']).\n\n"
            "Do not include opinions, tone analysis, or markdown."
        )

        user_prompt = "Analyze these video frames and describe the primary factual events."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    *image_contents
                ]
            }
        ]

        def _sdk_call() -> str:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        content = await asyncio.to_thread(_sdk_call)
        
        if not content:
            raise ValueError("No response from Azure OpenAI Vision for video frames")

        result = json.loads(content.strip())
        return {
            "scene_summary": result.get("scene_summary", "No visual events detected."),
            "events_detected": result.get("events_detected", [])
        }

    except Exception as e:
        print(f"❌ [Scene Describer] Analysis failed: {e}")
        return {
            "scene_summary": f"Visual analysis failed: {str(e)}",
            "events_detected": []
        }
