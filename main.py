import os
import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional, Any, TypedDict

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Type Definitions (similar to TypeScript interfaces) ---
class CommentaryOutput(TypedDict):
    general_comment: str
    comment_a: str
    comment_b: str
    comment_c: str
    comment_d: str
    comment_e: str

class ModelCommentProcessingResult(CommentaryOutput):
    processing_status: Literal['completed', 'failed']

class AnswerCommentsMap(TypedDict, total=False):
    openai: Optional[ModelCommentProcessingResult]
    claude: Optional[ModelCommentProcessingResult]
    gemini: Optional[ModelCommentProcessingResult]

# --- Configuration ---
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GROK_API_KEY = os.getenv('GROK_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
GROK_REASONING_EFFORT = os.getenv('GROK_REASONING_EFFORT', 'low')

# Rate limiting configuration
RATE_LIMITS = {
  "openai": {"max_concurrent": 5, "delay_s": 0.1},
  "claude": {"max_concurrent": 2, "delay_s": 0.5},
  "gemini": {"max_concurrent": 4, "delay_s": 0.15},
  "grok": {"max_concurrent": 3, "delay_s": 0.2},
  "mistral": {"max_concurrent": 3, "delay_s": 0.3}
}

API_TIMEOUT = 60  # 60 seconds

# --- Clients and Services ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase = None

http_client = httpx.AsyncClient()

# --- Rate Limiter Implementation ---
class RateLimiter:
    def __init__(self, max_concurrent: int, delay_s: float):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay_s = delay_s

    async def __aenter__(self):
        await self.semaphore.acquire()
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()

rate_limiters = {
    model: RateLimiter(config["max_concurrent"], config["delay_s"])
    for model, config in RATE_LIMITS.items()
}

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions ---
def log_supabase_error(context: str, error: Any, additional_data: Optional[Dict] = None):
    """Helper function to properly log Supabase errors."""
    error_details = {
        "message": getattr(error, 'message', str(error)),
        "details": getattr(error, 'details', None),
        "hint": getattr(error, 'hint', None),
        "code": getattr(error, 'code', None),
    }
    logger.error(f"{context}: {error_details}")
    if additional_data:
        logger.error(f"{context} - Additional Data: {additional_data}")


async def call_openai(prompt: str) -> Dict[str, str]:
    """OpenAI API call"""
    if not OPENAI_API_KEY:
        raise ValueError('OpenAI API key not configured')

    json_schema = {
        "name": "answer_comments",
        "schema": {
            "type": "object",
            "properties": {
                "general_comment": {"type": "string", "description": "Allgemeiner Kommentar zur Frage (max. 100 Wörter)"},
                "comment_a": {"type": "string", "description": "Kurzer Kommentar zu Antwort A (max. 50 Wörter)"},
                "comment_b": {"type": "string", "description": "Kurzer Kommentar zu Antwort B (max. 50 Wörter)"},
                "comment_c": {"type": "string", "description": "Kurzer Kommentar zu Antwort C (max. 50 Wörter)"},
                "comment_d": {"type": "string", "description": "Kurzer Kommentar zu Antwort D (max. 50 Wörter)"},
                "comment_e": {"type": "string", "description": "Kurzer Kommentar zu Antwort E (max. 50 Wörter)"},
            },
            "required": ["general_comment", "comment_a", "comment_b", "comment_c", "comment_d", "comment_e"],
        }
    }

    response = await http_client.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': 'o4-mini',
            'messages': [
                {'role': 'system', 'content': 'Du bist ein Experte für medizinische Prüfungsfragen, der strukturierte Kommentare zu Antwortmöglichkeiten auf Deutsch abgibt. Antworte immer auf Deutsch im JSON-Format.'},
                {'role': 'user', 'content': prompt}
            ],
            'response_format': {'type': "json_schema", 'json_schema': json_schema}
        },
        timeout=API_TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    return json.loads(data['choices'][0]['message']['content'])


async def call_gemini_base(prompt: str, model: str) -> Dict[str, str]:
    """Base function for Gemini API calls."""
    if not GEMINI_API_KEY:
        raise ValueError('Gemini API key not configured')

    gemini_schema = {
        "type": "object",
        "properties": {
            "general_comment": {"type": "string"},
            "comment_a": {"type": "string"},
            "comment_b": {"type": "string"},
            "comment_c": {"type": "string"},
            "comment_d": {"type": "string"},
            "comment_e": {"type": "string"},
        },
        "required": ["general_comment", "comment_a", "comment_b", "comment_c", "comment_d", "comment_e"]
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Du bist ein Experte für medizinische Prüfungsfragen. Antworte immer auf Deutsch.\nDeine Ausgabe muss ein String sein, der ein valides JSON-Objekt darstellt und dem Schema entspricht.\nAchte besonders darauf, dass alle Strings innerhalb des JSON (z.B. Kommentare) korrekt JSON-escaped sind (z.B. Zeilenumbrüche als \\n, Anführungszeichen als \\\").\n\n{prompt}"
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 5500,
            "temperature": 0.7,
            "responseMimeType": "application/json",
            "response_schema": gemini_schema
        }
    }
    
    response = await http_client.post(url, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    if not data.get('candidates'):
        raise ValueError('Gemini API Error: No candidates found in response')
    
    candidate = data['candidates'][0]
    if not candidate.get('content', {}).get('parts'):
        raise ValueError('Gemini API Error: No content parts found in candidate')
    
    text_content = candidate['content']['parts'][0].get('text')
    if not text_content:
        raise ValueError('Gemini API Error: No text found in the first part of the candidate')

    return json.loads(text_content)

async def call_gemini(prompt: str) -> Dict[str, str]:
    """Gemini 2.5 Pro API call"""
    return await call_gemini_base(prompt, "gemini-2.5-pro")

async def call_gemini_flash(prompt: str) -> Dict[str, str]:
    """Gemini 2.5 Flash API call"""
    return await call_gemini_base(prompt, "gemini-2.5-flash")

async def call_mistral(prompt: str) -> Dict[str, str]:
    """Mistral Magistral Small API call"""
    if not MISTRAL_API_KEY:
        raise ValueError('Mistral API key not configured')

    async with rate_limiters['mistral']:
        response = await http_client.post(
            'https://api.mistral.ai/v1/chat/completions',
            headers={'Authorization': f'Bearer {MISTRAL_API_KEY}', 'Content-Type': 'application/json'},
            json={
                'model': 'magistral-small-2506',
                'messages': [
                    {'role': 'system', 'content': 'Du bist ein Experte für medizinische Prüfungsfragen, der strukturierte Kommentare zu Antwortmöglichkeiten auf Deutsch abgibt. Antworte IMMER im JSON-Format.'},
                    {'role': 'user', 'content': prompt + '\n\nBitte antworte nur mit dem JSON-Objekt, ohne zusätzlichen Text oder Markdown-Formatierung.'}
                ],
                'response_format': {'type': 'json_object'},
                'temperature': 0.7,
                'max_tokens': 4096,
            },
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        content_str = data['choices'][0]['message']['content']
        
        parsed_content = json.loads(content_str)
        required_fields = ['general_comment', 'comment_a', 'comment_b', 'comment_c', 'comment_d', 'comment_e']
        for field in required_fields:
            if field not in parsed_content:
                parsed_content[field] = 'Keine Bewertung verfügbar.'
        return parsed_content


async def call_grok(prompt: str) -> Dict[str, str]:
    """Grok 3 Mini API call (via OpenRouter)"""
    if not OPENROUTER_API_KEY:
        raise ValueError('OpenRouter API key not configured')

    async with rate_limiters['grok']:
        response = await http_client.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://altfragen.com',
                'X-Title': 'AltFragen AI Commentary'
            },
            json={
                'model': 'x-ai/grok-3-mini',
                'messages': [
                     {'role': 'system', 'content': 'Du bist ein Experte für medizinische Prüfungsfragen, der strukturierte Kommentare zu Antwortmöglichkeiten auf Deutsch abgibt. Antworte IMMER im JSON-Format.'},
                    {'role': 'user', 'content': prompt + '\n\nBitte antworte nur mit dem JSON-Objekt, ohne zusätzlichen Text oder Markdown-Formatierung.'}
                ],
                'response_format': {'type': 'json_object'},
                'temperature': 0.7,
                'max_tokens': 4096,
            },
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        content_str = data['choices'][0]['message']['content']

        parsed_content = json.loads(content_str)
        required_fields = ['general_comment', 'comment_a', 'comment_b', 'comment_c', 'comment_d', 'comment_e']
        for field in required_fields:
            if field not in parsed_content:
                parsed_content[field] = 'Keine Bewertung verfügbar.'
        return parsed_content

async def generate_commentary(question: Dict, model_name: str) -> Optional[CommentaryOutput]:
    prompt = f"""Analysiere diese Multiple-Choice-Frage und erstelle kurze Kommentare für jede Antwortmöglichkeit:

Frage: {question.get('question')}
A) {question.get('option_a')}
B) {question.get('option_b')}
C) {question.get('option_c')}
D) {question.get('option_d')}
E) {question.get('option_e')}

Richtige Antwort aus Gedächtnisprotokoll: {question.get('correct_answer')}
Fachbereich: {question.get('subject')}
Kommentar aus Gedächtnisprotokoll: {question.get('comment')}

Erstelle:
1. Einen allgemeinen Kommentar zur Frage (max. 100 Wörter), der folgende Fragen beantwortet:
   - Was ist der Kerninhalt der Frage? Gib eine kurze, auf den Punkt gebrachte Übersicht.
   - Ist die protokollierte Lösung korrekt? Analysiere kritisch und wähle im Zweifel eine andere Antwort, wenn diese deutlich besser passt.
   - Warum ist die richtige Antwort korrekt?
   - Warum sind die anderen Antworten falsch?

2. Für jede Antwortmöglichkeit (A-E) einen kurzen, prägnanten Kommentar (max. 50 Wörter), der spezifisch erklärt warum diese Antwort richtig oder falsch ist."""

    model_key = model_name.lower()
    
    async with rate_limiters[model_key]:
        try:
            if model_key == 'openai':
                return await call_openai(prompt)
            
            elif model_key == 'claude':  # Note: 'claude' now maps to Grok -> Mistral fallback
                try:
                    return await call_grok(prompt)
                except Exception as grok_error:
                    logger.warning(f"Grok failed, falling back to Mistral. Error: {grok_error}")
                    try:
                        return await call_mistral(prompt)
                    except Exception as mistral_error:
                        logger.error(f"Mistral fallback also failed: {mistral_error}")
                        raise Exception(f"Both Grok and Mistral failed. Grok: {grok_error}, Mistral: {mistral_error}")

            elif model_key == 'gemini':
                try:
                    return await call_gemini(prompt)
                except Exception as gemini_error:
                    error_msg = str(gemini_error).lower()
                    is_quota_error = '429' in error_msg or 'quota' in error_msg
                    if is_quota_error:
                        logger.warning("Gemini Pro quota exceeded, falling back to Gemini Flash.")
                        try:
                            return await call_gemini_flash(prompt)
                        except Exception as flash_error:
                            logger.error(f"Gemini Flash fallback also failed: {flash_error}")
                            raise Exception(f"Both Gemini Pro and Flash failed. Pro: {gemini_error}, Flash: {flash_error}")
                    else:
                        raise gemini_error
            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            logger.error(f"Error generating commentary for {model_name}: {e}")
            raise

async def generate_summary(commentaries: AnswerCommentsMap, question: Dict) -> Optional[CommentaryOutput]:
    if not OPENAI_API_KEY:
        raise ValueError('OpenAI API key not configured for summary generation')

    model_names = commentaries.keys()
    general_comments = "\n\n".join([f"{model.upper()}: {commentaries[model]['general_comment']}" for model in model_names])
    answer_comments = "\n\n".join(
        [f"Antwort {letter.upper()}:\n" + "\n".join(
            [f"{model.upper()}: {commentaries[model][f'comment_{letter}']}" for model in model_names]
        ) for letter in ['a', 'b', 'c', 'd', 'e']]
    )

    summary_prompt = f"""Basierend auf den folgenden KI-Kommentaren zu einer {question.get('subject')}-Frage, erstelle eine strukturierte Zusammenfassung mit Übereinstimmungsanalyse:

Frage: {question.get('question')}
Richtige Antwort: {question.get('correct_answer')}

ALLGEMEINE KOMMENTARE:
{general_comments}

ANTWORT-SPEZIFISCHE KOMMENTARE:
{answer_comments}

Erstelle:
1. Einen allgemeinen Zusammenfassungskommentar (max. 150 Wörter), der die wichtigsten Erkenntnisse synthetisiert
2. Für jede Antwortmöglichkeit (A-E) einen kurzen Zusammenfassungskommentar (max. 50 Wörter)

Antworte auf Deutsch."""

    json_schema = {
        "name": "summary_comments",
        "schema": {
            "type": "object",
            "properties": {
                "general_comment": {"type": "string", "description": "Allgemeiner Zusammenfassungskommentar (max. 150 Wörter)"},
                "comment_a": {"type": "string", "description": "Zusammenfassungskommentar zu Antwort A (max. 50 Wörter)"},
                "comment_b": {"type": "string", "description": "Zusammenfassungskommentar zu Antwort B (max. 50 Wörter)"},
                "comment_c": {"type": "string", "description": "Zusammenfassungskommentar zu Antwort C (max. 50 Wörter)"},
                "comment_d": {"type": "string", "description": "Zusammenfassungskommentar zu Antwort D (max. 50 Wörter)"},
                "comment_e": {"type": "string", "description": "Zusammenfassungskommentar zu Antwort E (max. 50 Wörter)"},
            },
            "required": ["general_comment", "comment_a", "comment_b", "comment_c", "comment_d", "comment_e"],
        }
    }

    response = await http_client.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': 'gpt-4.1-nano',
            'messages': [
                {'role': 'system', 'content': 'Du bist eine Experten-KI für Bildung, die strukturierte Zusammenfassungen aus mehreren KI-Kommentaren erstellt. Antworte immer auf Deutsch im JSON-Format.'},
                {'role': 'user', 'content': summary_prompt}
            ],
            'response_format': {"type": "json_schema", "json_schema": json_schema}
        },
        timeout=API_TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    return json.loads(data['choices'][0]['message']['content'])


async def process_question(question: Dict, enabled_models: List[str]) -> Dict:
    """Processes a single question: generates commentaries and summaries, and updates the database."""
    try:
        await supabase.from_('questions').update({'ai_commentary_status': 'processing'}).eq('id', question['id']).execute()
        
        logger.info(f"Processing question {question['id']} with models: {enabled_models}")

        # Generate commentaries in parallel
        model_tasks = [generate_commentary(question, model) for model in enabled_models]
        model_results = await asyncio.gather(*model_tasks, return_exceptions=True)

        answer_comments: AnswerCommentsMap = {}
        for i, result in enumerate(model_results):
            model_name = enabled_models[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to get commentary from {model_name} for Q {question['id']}: {result}")
                answer_comments[model_name] = {'processing_status': 'failed', 'general_comment': f'Fehler: {result}', 'comment_a': f'Fehler: {result}', 'comment_b': f'Fehler: {result}', 'comment_c': f'Fehler: {result}', 'comment_d': f'Fehler: {result}', 'comment_e': f'Fehler: {result}'}
            else:
                answer_comments[model_name] = {'processing_status': 'completed', **result}
        
        # Insert answer comments
        if any(v['processing_status'] == 'completed' for v in answer_comments.values()):
            insert_data = {'question_id': question['id'], 'processing_status': 'completed'}
            for model, comments in answer_comments.items():
                if comments:
                    insert_data[f'{model}_general_comment'] = comments.get('general_comment')
                    for letter in "abcde":
                        insert_data[f'{model}_comment_{letter}'] = comments.get(f'comment_{letter}')

            res = await supabase.from_('ai_answer_comments').upsert(insert_data).execute()
            if hasattr(res, 'error') and res.error is not None:
                log_supabase_error("Error inserting answer comments", res.error, insert_data)
                raise Exception("Failed to insert comments")

            # Generate and insert summary
            successful_commentaries = {k: v for k, v in answer_comments.items() if v['processing_status'] == 'completed'}
            if successful_commentaries:
                try:
                    summary = await generate_summary(successful_commentaries, question)
                    if summary:
                        summary_data = {'question_id': question['id'], **{f'summary_{k}': v for k, v in summary.items()}}
                        res = await supabase.from_('ai_commentary_summaries').upsert(summary_data).execute()
                        if hasattr(res, 'error') and res.error is not None:
                            log_supabase_error("Error inserting summary", res.error, summary_data)
                except Exception as e:
                    logger.error(f"Error generating summary for Q {question['id']}: {e}")

        # Update status to completed
        await supabase.from_('questions').update({
            'ai_commentary_status': 'completed',
            'ai_commentary_processed_at': datetime.now(timezone.utc).isoformat()
        }).eq('id', question['id']).execute()
        
        logger.info(f"Successfully processed question {question['id']}")
        return {"success": True, "questionId": question['id']}

    except Exception as error:
        logger.error(f"Error processing question {question['id']}: {error}")
        await supabase.from_('questions').update({'ai_commentary_status': 'failed'}).eq('id', question['id']).execute()
        return {"success": False, "questionId": question['id'], "error": str(error)}


@app.post("/process-ai-commentary")
async def process_ai_commentary_endpoint():
    """
    Endpoint to trigger the AI commentary processing.
    This replaces the Deno `serve` function.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    logger.info("AI Commentary processing started")
    
    # 1. Get AI commentary settings
    try:
        res = await supabase.from_('ai_commentary_settings').select('*').single().execute()
        if hasattr(res, 'error') and res.error is not None:
             raise Exception(res.error)
        settings = res.data
    except Exception as e:
        log_supabase_error("Error fetching settings", e)
        raise HTTPException(status_code=500, detail="Settings not found")

    ai_settings = {
        'feature_enabled': settings.get('feature_enabled', False),
        'batch_size': settings.get('batch_size', 5),
        'processing_delay_minutes': settings.get('processing_delay_minutes', 60),
        'models_enabled': [model for model, enabled in settings.get('models_enabled', {}).items() if enabled]
    }
    logger.info(f"Settings loaded: {ai_settings}")

    if not ai_settings['feature_enabled']:
        logger.info("AI Commentary feature is disabled")
        return JSONResponse(content={"message": "AI Commentary feature is disabled", "processed": 0})

    # 2. Get existing question IDs to avoid reprocessing
    delay_threshold = datetime.now(timezone.utc) - timedelta(minutes=ai_settings['processing_delay_minutes'])
    
    res_comments = await supabase.from_('ai_answer_comments').select('question_id').execute()
    res_summaries = await supabase.from_('ai_commentary_summaries').select('question_id').execute()
    
    existing_question_ids = set()
    if hasattr(res_comments, 'data') and res_comments.data is not None:
        existing_question_ids.update(item['question_id'] for item in res_comments.data)
    if hasattr(res_summaries, 'data') and res_summaries.data is not None:
        existing_question_ids.update(item['question_id'] for item in res_summaries.data)

    logger.info(f"Found {len(existing_question_ids)} questions that already have commentary or summaries")

    # 3. Get pending questions
    try:
        res_questions = await supabase.from_('questions')\
            .select('*')\
            .eq('ai_commentary_status', 'pending')\
            .lt('ai_commentary_queued_at', delay_threshold.isoformat())\
            .limit(ai_settings['batch_size'] * 3)\
            .execute()
        if hasattr(res_questions, 'error') and res_questions.error is not None:
            raise Exception(res_questions.error)
        all_pending_questions = res_questions.data or []
    except Exception as e:
        log_supabase_error("Error fetching pending questions", e)
        raise HTTPException(status_code=500, detail="Failed to fetch questions")
    
    pending_questions = [
        q for q in all_pending_questions if q['id'] not in existing_question_ids
    ][:ai_settings['batch_size']]

    logger.info(f"Filtered down to {len(pending_questions)} questions that need processing")
    if not pending_questions:
        return JSONResponse(content={"message": "No questions to process", "processed": 0})

    # 4. Process questions in batches
    MAX_CONCURRENT_QUESTIONS = 3
    processed_count = 0
    
    question_batches = [
        pending_questions[i:i + MAX_CONCURRENT_QUESTIONS] 
        for i in range(0, len(pending_questions), MAX_CONCURRENT_QUESTIONS)
    ]
    
    for i, batch in enumerate(question_batches):
        logger.info(f"Processing batch {i + 1}/{len(question_batches)} ({len(batch)} questions)")
        tasks = [process_question(q, ai_settings['models_enabled']) for q in batch]
        results = await asyncio.gather(*tasks)
        
        batch_success_count = sum(1 for r in results if r['success'])
        processed_count += batch_success_count
        logger.info(f"Batch {i + 1} completed: {batch_success_count}/{len(batch)} successful")

    logger.info(f"AI Commentary processing completed. Processed: {processed_count} out of {len(pending_questions)}")
    return JSONResponse(
        content={"message": "Processing completed", "processed": processed_count}
    )

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Commentary Processor is running."}

if __name__ == "__main__":
    if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY]):
        logger.error("Supabase URL and Key must be set in environment variables.")
    else:
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port) 
