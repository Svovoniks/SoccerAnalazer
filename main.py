from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import os
from src.detection_pipeline import DetectionPipeline
from src.report_generator import generate_statistics_pdf
import json
from datetime import datetime
import uuid

app = FastAPI(title="Soccer Analysis API")

REQUEST_HISTORY_DIR = "request_history"
os.makedirs(REQUEST_HISTORY_DIR, exist_ok=True)

# Initialize the detection pipeline
model_path = os.path.join(os.path.dirname(__file__), 'src', 'model', 'best.pt')
detection_pipeline = DetectionPipeline(model_path)

# Load HTML template from file
def load_html_template():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return load_html_template()


@app.get("/results", response_class=HTMLResponse)
async def serve_results():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "results.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    try:
        # Generate UUID and timestamp for this request
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Save uploaded file to request_history with UUID only
        original_filename = f"{request_id}_in.mp4"
        original_path = os.path.join(REQUEST_HISTORY_DIR, original_filename)
        
        with open(original_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the video
        processed_filename = f"{request_id}_out.mp4"
        processed_path = os.path.join(REQUEST_HISTORY_DIR, processed_filename)
        
        # Process video with statistics
        frames, stats = detection_pipeline.detect_in_video_with_stats(original_path, processed_path)
        
        # Save statistics to request_history
        stats_filename = f"{request_id}_statistics.json"
        stats_path = os.path.join(REQUEST_HISTORY_DIR, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        # Generate and save PDF report automatically
        pdf_bytes = generate_statistics_pdf(stats.to_dict(), file.filename)
        report_filename = f"{request_id}_report.pdf"
        report_path = os.path.join(REQUEST_HISTORY_DIR, report_filename)
        with open(report_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Update request history mapping
        request_history_path = os.path.join(REQUEST_HISTORY_DIR, "request_history.json")
        request_history = {}
        if os.path.exists(request_history_path):
            with open(request_history_path, 'r') as f:
                request_history = json.load(f)
        
        request_history[request_id] = {
            "timestamp": timestamp,
            "original_filename": original_filename,
            "processed_filename": processed_filename,
            "statistics_filename": stats_filename,
            "report_filename": report_filename,
            "original_upload_name": file.filename
        }
        
        with open(request_history_path, 'w') as f:
            json.dump(request_history, f, indent=2)
        
        return {
            "status": "success",
            "message": "Video processed successfully",
            "request_id": request_id,
            "result_filename": processed_filename,
            "original_filename": original_filename,
            "statistics_filename": stats_filename,
            "report_filename": report_filename,
            "statistics": stats.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    # Try request_history first, then results for backward compatibility
    file_path = os.path.join(REQUEST_HISTORY_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/x-msvideo"
    )


@app.post("/generate-pdf")
async def generate_pdf_report(data: dict):
    try:
        request_id = data.get('request_id')
        
        if not request_id:
            raise HTTPException(status_code=400, detail="No request_id provided")
        
        # Load request history to find the report filename
        request_history_path = os.path.join(REQUEST_HISTORY_DIR, "request_history.json")
        if not os.path.exists(request_history_path):
            raise HTTPException(status_code=404, detail="No request history found")
        
        with open(request_history_path, 'r') as f:
            request_history = json.load(f)
        
        if request_id not in request_history:
            raise HTTPException(status_code=404, detail="Request ID not found")
        
        report_filename = request_history[request_id].get('report_filename')
        if not report_filename:
            raise HTTPException(status_code=404, detail="No report found for this request")
        
        report_path = os.path.join(REQUEST_HISTORY_DIR, report_filename)
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return {
            "status": "success",
            "report_filename": report_filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
