"""Mitesh AI Coach - Pipecat Cloud Voice Bot with RAG (v6)"""

import os
import json
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

import openai as openai_module
from supabase import create_client

load_dotenv(override=True)

logger.info("Mitesh Bot v6.0-RAG starting...")

# ──────────────────────────── Supabase ────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase connected")
else:
    logger.warning("Supabase credentials missing - RAG disabled")

oai_client = openai_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_PROFILE_ID = os.getenv("DEFAULT_PROFILE_ID", None)
_current_profile_id = None


# ──────────────────────────── RAG Functions ────────────────────────────
def fetch_knowledge_sync(query_text, profile_id=None):
    if not supabase or not query_text.strip():
        return "No knowledge available."
    try:
        embedding_response = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
        )
        query_embedding = embedding_response.data[0].embedding

        if profile_id:
            try:
                result = supabase.rpc("match_knowledge", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.35,
                    "match_count": 5,
                    "p_profile_id": profile_id,
                }).execute()
                if result.data and len(result.data) > 0:
                    chunks = [f"[{c.get('source_title', 'Source')}]: {c.get('content', '')}" for c in result.data]
                    logger.info(f"RAG: Found {len(result.data)} chunks via match_knowledge")
                    return "\n\n".join(chunks)
            except Exception as e:
                logger.warning(f"match_knowledge failed: {e}")

        try:
            result = supabase.rpc("match_knowledge_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 3,
            }).execute()
            if result.data and len(result.data) > 0:
                chunks = [c.get("content", "") for c in result.data]
                logger.info(f"RAG: Found {len(result.data)} chunks via match_knowledge_chunks")
                return "\n\n".join(chunks)
        except Exception as e:
            logger.warning(f"match_knowledge_chunks failed: {e}")

        return "No relevant knowledge found."
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return "Knowledge search failed."


def get_profile_info_sync(profile_id):
    if not supabase or not profile_id:
        return {}
    try:
        result = supabase.from_("mind_profile").select(
            "name, headline, description, purpose, instructions, speaking_style"
        ).eq("id", profile_id).single().execute()
        if result.data:
            logger.info(f"Profile loaded: {result.data.get('name', 'Unknown')}")
            return result.data
    except Exception as e:
        logger.warning(f"Profile fetch failed: {e}")
    return {}


# ──────────────────────────── Function Handler (Official Pattern) ────────────────────────────
async def handle_search_knowledge(params: FunctionCallParams):
    """Official Pipecat function call handler pattern.
    MUST call params.result_callback() to return data to LLM."""
    global _current_profile_id
    try:
        query = params.arguments.get("query", "")
        logger.info(f"Function call: search_knowledge_base('{query}')")
        knowledge = fetch_knowledge_sync(query, _current_profile_id)
        logger.info(f"Knowledge result: {len(knowledge)} chars")
        await params.result_callback({"knowledge": knowledge})
    except Exception as e:
        logger.error(f"Function call error: {e}")
        await params.result_callback({"error": str(e)})


# ──────────────────────────── Tools ────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for information about coaching, Law of Attraction, NLP, meditation, motivation. Use this for any coaching question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ──────────────────────────── Transport ────────────────────────────
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


# ──────────────────────────── Bot ────────────────────────────
async def run_bot(transport: BaseTransport, _runner_args):
    global _current_profile_id

    logger.info("Starting pipeline v6...")

    profile_id = DEFAULT_PROFILE_ID
    if hasattr(_runner_args, 'body') and _runner_args.body:
        body = _runner_args.body if isinstance(_runner_args.body, dict) else {}
        profile_id = body.get("profile_id", body.get("profileId", DEFAULT_PROFILE_ID))

    _current_profile_id = profile_id
    logger.info(f"Profile ID: {profile_id}")

    profile = get_profile_info_sync(profile_id) if profile_id else {}
    profile_name = profile.get("name", "Mitesh Khatri")
    profile_headline = profile.get("headline", "Law of Attraction Coach")
    profile_description = profile.get("description", "A renowned life coach.")
    profile_style = profile.get("speaking_style", "Warm, energetic, high-vibe.")

    system_prompt = f"""You are an AI voice clone of {profile_name}, {profile_headline}.
Biography: {profile_description}
Speaking Style: {profile_style}

RULES:
- LIVE VOICE CALL. Keep responses to 2-3 sentences max.
- Speak naturally like a real person on a phone call.
- Use casual Hinglish (Hindi + English mix).
- Be warm and encouraging. Use "Hey Champion", "Bilkul", "Dekho", "Acha".
- ALWAYS call search_knowledge_base BEFORE answering coaching questions.
- After getting knowledge, explain simply in your own words.
- NEVER read URLs, links, or use markdown formatting.
- Talk like chatting with a friend."""

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="whisper-1",
        language="hi",
    )
    logger.info("STT: gpt-4o-transcribe (Hindi)")

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    logger.info("LLM: gpt-4o-mini")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        model_id="sonic-multilingual",
    )
    logger.info(f"TTS: Cartesia voice={os.getenv('CARTESIA_VOICE_ID', 'NOT SET')}")

    # Register function - official pattern with FunctionCallParams
    llm.register_function(
        "search_knowledge_base",
        handle_search_knowledge,
        cancel_on_interruption=True,
    )
    logger.info("Function search_knowledge_base registered")

    messages = [{"role": "system", "content": system_prompt}]
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({
            "role": "system",
            "content": "Say hello warmly. Introduce yourself as Mitesh Khatri in 2 sentences max."
        })
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    logger.info("Pipeline ready")
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args):
    from pipecat.runner.utils import create_transport
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()