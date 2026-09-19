import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import httpx
from rembg import new_session, remove

app = FastAPI()

session = new_session(model_name="birefnet-general")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Jewelry White Background Precision API"
    }

@app.get("/process-white-bg")
async def process_white_bg(image_url: str):
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing image_url parameter")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(image_url, timeout=25.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Unable to fetch source image")

        input_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        cutout = remove(
            input_img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=220,
            alpha_matting_background_threshold=15,
            alpha_matting_erode_size=1,
            post_process_mask=True,
        )

        cutout_np = np.array(cutout)
        rgb = cutout_np[:, :, :3]
        alpha = cutout_np[:, :, 3]

        alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
        white_canvas = np.ones_like(rgb, dtype=np.float32) * 255.0
        blended = (rgb.astype(np.float32) * alpha_factor) + (white_canvas * (1.0 - alpha_factor))

        final_img = Image.fromarray(blended.astype(np.uint8))

        output = io.BytesIO()
        final_img.save(output, format="JPEG", quality=95, optimize=True)
        return Response(content=output.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
