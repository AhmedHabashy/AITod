# Video Transcriber - Project Plan

## Project Overview
A web-based video transcription and translation service that extracts audio from videos, transcribes it with timestamps, and generates professionally translated SRT subtitle files using context-aware LLM translation.

## Core Features
1. Video upload through web interface
2. Language selection for transcription/translation
3. Audio extraction with precise timestamps
4. LLM-powered audio transcription (Gemini/GPT)
5. Context-aware professional translation
6. SRT file generation with timestamps and translations

---

## Technology Stack

### Frontend
- **Framework**: React.js or Next.js
- **UI Library**: Tailwind CSS / Material-UI
- **File Upload**: React Dropzone
- **HTTP Client**: Axios

### Backend
- **API Framework**: FastAPI (Python)
- **ASGI Server**: Uvicorn
- **File Processing**:
  - FFmpeg (audio extraction)
  - MoviePy (video processing alternative)
- **LLM Integration**:
  - OpenAI API (Whisper for transcription)
  - Google Gemini API (alternative)
  - OpenAI GPT-4 (translation with context)

### Storage
- **Temporary Storage**: Local file system
- **Database** (optional): SQLite/PostgreSQL for tracking jobs
- **File Formats**:
  - Input: MP4, AVI, MOV, MKV
  - Intermediate: WAV/MP3 (audio), CSV (timestamps)
  - Output: SRT (subtitles)

### Additional Libraries
- **pydub**: Audio manipulation
- **srt**: SRT file generation
- **pandas**: CSV handling
- **python-multipart**: File uploads in FastAPI
- **python-dotenv**: Environment configuration

---

## System Architecture

```
┌─────────────────┐
│   Web Frontend  │
│   (React/Next)  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│    FastAPI      │
│   Backend API   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────┐
│ FFmpeg │  │ LLM APIs │
│ Audio  │  │ Gemini/  │
│Extract │  │ GPT-4    │
└────────┘  └──────────┘
    │            │
    ▼            ▼
┌─────────────────────┐
│  Processing Pipeline│
│ 1. Extract Audio    │
│ 2. Transcribe       │
│ 3. Build Context    │
│ 4. Translate        │
│ 5. Generate SRT     │
└─────────────────────┘
```

---

## File Structure

```
Video_Transcriber/
│
├── frontend/                    # Web Interface
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoUploader.jsx
│   │   │   ├── LanguageSelector.jsx
│   │   │   ├── ProgressTracker.jsx
│   │   │   └── DownloadButton.jsx
│   │   ├── pages/
│   │   │   └── index.jsx
│   │   ├── api/
│   │   │   └── client.js
│   │   └── App.jsx
│   ├── package.json
│   └── README.md
│
├── backend/                     # FastAPI Backend
│   ├── app/
│   │   ├── main.py             # FastAPI app entry
│   │   ├── config.py           # Configuration
│   │   ├── models/
│   │   │   ├── request.py      # Request models
│   │   │   └── response.py     # Response models
│   │   ├── services/
│   │   │   ├── audio_extractor.py
│   │   │   ├── transcriber.py
│   │   │   ├── translator.py
│   │   │   ├── context_builder.py
│   │   │   └── srt_generator.py
│   │   ├── api/
│   │   │   └── routes.py       # API endpoints
│   │   └── utils/
│   │       ├── file_handler.py
│   │       └── llm_client.py
│   ├── requirements.txt
│   └── .env.example
│
├── storage/                     # Temporary file storage
│   ├── uploads/                # Uploaded videos
│   ├── audio/                  # Extracted audio
│   ├── transcripts/            # CSV transcripts
│   └── output/                 # Generated SRT files
│
├── tests/                      # Unit tests
│   ├── test_audio_extractor.py
│   ├── test_transcriber.py
│   └── test_translator.py
│
├── .env                        # Environment variables
├── .gitignore
├── docker-compose.yml          # Optional containerization
├── PROJECT_PLAN.md            # This file
└── README.md                   # Project documentation
```

---

## Detailed Workflow

### Phase 1: Video Upload & Setup
1. User accesses web interface
2. User uploads video file (drag-and-drop or file picker)
3. User selects source language and target translation language
4. Frontend sends video + metadata to FastAPI endpoint

### Phase 2: Audio Extraction
1. Backend receives video file
2. Save video to `storage/uploads/` with unique ID
3. Use FFmpeg to extract audio:
   - Extract audio track
   - Convert to WAV format for better processing
   - Maintain sync with video timestamps
4. Save audio to `storage/audio/{video_id}.wav`

### Phase 3: Transcription
1. Use OpenAI Whisper or Gemini API for audio transcription
2. Get timestamped segments:
   ```
   [
     {start: 0.0, end: 2.5, text: "Hello world"},
     {start: 2.5, end: 5.0, text: "This is a test"}
   ]
   ```
3. Save to CSV: `storage/transcripts/{video_id}.csv`
   ```csv
   start_time,end_time,text
   0.0,2.5,"Hello world"
   2.5,5.0,"This is a test"
   ```

### Phase 4: Context Building
1. Read entire CSV transcript
2. Build global context (summary of full content):
   - Extract key topics
   - Identify domain/subject matter
   - Note any technical terms or proper nouns
3. Store context as string (500-1000 tokens)

### Phase 5: Context-Aware Translation
1. For each transcript segment:
   ```python
   prompt = f"""
   Context: {global_context}

   Source Language: {source_lang}
   Target Language: {target_lang}

   Translate the following professionally:
   "{segment_text}"

   Keep cultural nuances and maintain professional tone.
   """
   ```
2. Send to GPT-4/Gemini for translation
3. Store translated segments with timestamps

### Phase 6: SRT Generation
1. Convert translated segments to SRT format:
   ```
   1
   00:00:00,000 --> 00:00:02,500
   Translated text here

   2
   00:00:02,500 --> 00:00:05,000
   Next translated segment
   ```
2. Save to `storage/output/{video_id}.srt`
3. Return download link to frontend

---

## API Endpoints

### POST /api/upload
- **Description**: Upload video file
- **Request**: Multipart form data (video file)
- **Response**: `{video_id: string, status: "uploaded"}`

### POST /api/transcribe
- **Description**: Start transcription process
- **Request**:
  ```json
  {
    "video_id": "uuid",
    "source_language": "en",
    "target_language": "es"
  }
  ```
- **Response**: `{job_id: string, status: "processing"}`

### GET /api/status/{job_id}
- **Description**: Check transcription progress
- **Response**:
  ```json
  {
    "status": "processing|completed|failed",
    "progress": 75,
    "current_step": "translating"
  }
  ```

### GET /api/download/{job_id}
- **Description**: Download SRT file
- **Response**: SRT file download

### GET /api/languages
- **Description**: Get supported languages
- **Response**: `{languages: [{code: "en", name: "English"}, ...]}`

---

## Implementation Phases

### Phase 1: Project Setup (Week 1)
- [ ] Initialize frontend (React/Next.js)
- [ ] Setup FastAPI backend structure
- [ ] Configure virtual environment
- [ ] Install core dependencies
- [ ] Setup environment variables (.env)
- [ ] Create folder structure

### Phase 2: Backend Core (Week 2)
- [ ] Implement file upload endpoint
- [ ] Build audio extraction service (FFmpeg)
- [ ] Integrate LLM API (Whisper/Gemini)
- [ ] Create transcription service
- [ ] Implement CSV storage for timestamps

### Phase 3: Translation Pipeline (Week 3)
- [ ] Build context builder
- [ ] Implement context-aware translation
- [ ] Create SRT generator
- [ ] Add error handling and retries
- [ ] Optimize API calls (batching if possible)

### Phase 4: Frontend Development (Week 4)
- [ ] Create video upload component
- [ ] Build language selector
- [ ] Implement progress tracker
- [ ] Add download functionality
- [ ] Connect to backend API

### Phase 5: Testing & Optimization (Week 5)
- [ ] Write unit tests for services
- [ ] Test with various video formats
- [ ] Optimize processing speed
- [ ] Handle large files efficiently
- [ ] Add input validation

### Phase 6: Polish & Deployment (Week 6)
- [ ] Add error messages and user feedback
- [ ] Improve UI/UX
- [ ] Add logging and monitoring
- [ ] Create documentation
- [ ] Optional: Docker containerization

---

## Key Considerations

### Performance Optimization
- Process audio in chunks for large videos
- Use async processing for API calls
- Implement caching for repeated translations
- Queue system for multiple concurrent jobs

### Error Handling
- Validate video file format and size
- Handle API rate limits gracefully
- Implement retry logic for failed API calls
- Provide clear error messages to users

### Security
- Sanitize file uploads
- Set maximum file size limits
- Secure API keys in environment variables
- Implement request rate limiting
- Clean up temporary files after processing

### Cost Management
- Monitor LLM API usage
- Implement token counting
- Set per-user or per-day limits
- Cache translations for common phrases

---

## Environment Variables (.env)

```env
# API Keys
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# File Storage
MAX_FILE_SIZE_MB=500
UPLOAD_DIR=./storage/uploads
AUDIO_DIR=./storage/audio
TRANSCRIPT_DIR=./storage/transcripts
OUTPUT_DIR=./storage/output

# Processing
CHUNK_SIZE_SECONDS=30
MAX_CONCURRENT_JOBS=5

# Frontend
FRONTEND_URL=http://localhost:3000
```

---

## Success Metrics
- Successfully process videos up to 500MB
- Transcription accuracy > 90%
- Translation maintains context and professionalism
- End-to-end processing time < 5 minutes for 10-minute video
- Support for 10+ languages

---

## Future Enhancements
- Support for multiple subtitle tracks
- Real-time preview of subtitles on video
- Batch processing for multiple videos
- User accounts and history
- Direct integration with YouTube/Vimeo
- Support for audio-only files
- Custom vocabulary/terminology for specific domains
- Export in multiple formats (VTT, ASS, TXT)

---

## Getting Started

1. Clone/setup the repository
2. Create virtual environment: `python -m venv .venv`
3. Install backend dependencies: `pip install -r backend/requirements.txt`
4. Install frontend dependencies: `cd frontend && npm install`
5. Configure `.env` file with API keys
6. Start backend: `uvicorn backend.app.main:app --reload`
7. Start frontend: `cd frontend && npm run dev`
8. Access application at `http://localhost:3000`

---

*Last Updated: 2026-01-06*
*Status: Planning Phase*
