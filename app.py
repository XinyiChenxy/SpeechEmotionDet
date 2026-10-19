"""English local UI + FastAPI API. Start with ./run_app.sh."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
from typing import Callable, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ser.app_service import BEST_CHECKPOINT, MAX_SECONDS, MAX_UPLOAD_BYTES, BestModelPredictor, best_checkpoint_info
from ser.common import LABELS, ROOT

logger = logging.getLogger('speech-emotion-app')
EXAMPLES = {
    'clip-01': ('Sample 01', ROOT / 'data/AudioWAV/1001_DFA_ANG_XX.wav'),
    'clip-02': ('Sample 02', ROOT / 'data/AudioWAV/1001_DFA_NEU_XX.wav'),
    'clip-03': ('Sample 03', ROOT / 'data/AudioWAV/1001_DFA_SAD_XX.wav'),
}


class Prediction(BaseModel):
    filename: str
    prediction: Literal['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    probabilities: dict[str, float]
    duration_seconds: float
    inference_seconds: float
    best_epoch: int
    checkpoint_sha256: str
    encoder: str
    head: str


def create_app(predictor_factory: Callable = BestModelPredictor,
               info_provider: Callable = best_checkpoint_info):
    @asynccontextmanager
    async def lifespan(application):
        yield
        application.state.predictor = None

    application = FastAPI(title='Speech Emotion Lab', version='1.0.0', lifespan=lifespan,
                          description='Frozen emotion2vec features and the validation-selected CREMA-D classifier.')
    application.state.predictor = None
    application.state.load_lock = threading.Lock()
    application.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')

    def get_predictor():
        with application.state.load_lock:
            if application.state.predictor is None:
                application.state.predictor = predictor_factory()
            return application.state.predictor

    @application.middleware('http')
    async def check_upload_size(request: Request, call_next):
        if request.url.path == '/api/predict':
            try:
                length = int(request.headers.get('content-length', '0'))
            except ValueError:
                return JSONResponse({'detail': 'Invalid Content-Length header.'}, status_code=400)
            # Allow a small multipart envelope; the file itself is checked below.
            if length > MAX_UPLOAD_BYTES + 1024 * 1024:
                return JSONResponse({'detail': 'The audio file exceeds the 20 MB limit.'}, status_code=413)
        return await call_next(request)

    @application.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'web/index.html', media_type='text/html')

    @application.get('/health')
    def health():
        return {'status': 'ok', 'model_loaded': application.state.predictor is not None}

    @application.get('/api/model')
    def model():
        try:
            current = application.state.predictor
            info = current.info if current is not None else info_provider()
        except (OSError, ValueError, RuntimeError):
            logger.exception('Best checkpoint metadata could not be loaded')
            raise HTTPException(503, 'The best model is not ready. Check the model files and server log.')
        return {key: info[key] for key in ('best_epoch', 'val_macro_f1', 'encoder', 'head', 'selection', 'labels')} | {
            'checkpoint_name': 'best.pt', 'max_upload_mb': MAX_UPLOAD_BYTES // (1024 * 1024),
            'max_seconds': MAX_SECONDS,
        }

    @application.get('/api/examples')
    def examples():
        return [{'id': key, 'title': title, 'filename': path.name, 'audio_url': f'/api/examples/{key}/audio'}
                for key, (title, path) in EXAMPLES.items() if path.is_file()]

    @application.get('/api/examples/{example_id}/audio')
    def example_audio(example_id: str):
        if example_id not in EXAMPLES or not EXAMPLES[example_id][1].is_file():
            raise HTTPException(404, 'This audio sample is not available.')
        path = EXAMPLES[example_id][1]
        return FileResponse(path, media_type='audio/wav', filename=path.name, content_disposition_type='inline')

    @application.post('/api/predict', response_model=Prediction)
    def predict(request: Request, file: UploadFile = File(...)):
        try:
            filename = file.filename or ''
            if Path(filename).suffix.lower() not in ('.wav', '.mp3'):
                raise HTTPException(400, 'Please select a WAV or MP3 audio file.')
            content = file.file.read(MAX_UPLOAD_BYTES + 1)
            if not content:
                raise HTTPException(400, 'The audio file is empty. Please select another file.')
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, 'The audio file exceeds the 20 MB limit.')
            try:
                predictor = get_predictor()
            except (OSError, ValueError, RuntimeError, ImportError):
                logger.exception('Best model initialization failed')
                raise HTTPException(503, 'The saved best model could not be loaded. Check the server log.')
            try:
                return predictor.predict_bytes(content, filename)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
            except (RuntimeError, FloatingPointError, OSError):
                logger.exception('Audio inference failed')
                raise HTTPException(500, 'The audio could not be analyzed. Try a shorter clip or check the server log.')
        finally:
            file.file.close()

    return application


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='127.0.0.1', port=8000)
