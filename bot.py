"""Mitesh AI Coach - Pipecat Cloud Voice Bot with RAG (v6.4 Language Fix)"""

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

logger.info("Mitesh Bot v6.4-LANGUAGE-FIX starting...")

# ──────────────────────────── Config ────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HARDCODED_PROFILE_ID = "1cb7dee0-815f-4278-b93e-062bdf486389"

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase connected")
else:
    logger.warning("Supabase credentials missing")

oai_client = openai_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ──────────────────────────── RAG ────────────────────────────
def fetch_knowledge_sync(query_text):
    """Fetch knowledge from Supabase. Always uses hardcoded profile ID."""
    if not supabase or not query_text.strip():
        logger.warning("RAG skip: no supabase or empty query")
        return "No knowledge available."

    try:
        # Step 1: Generate embedding
        logger.info(f"RAG: Generating embedding for: '{query_text}'")
        embedding_response = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
        )
        query_embedding = embedding_response.data[0].embedding
        logger.info(f"RAG: Embedding generated, length={len(query_embedding)}")

        # Step 2: Call match_knowledge with hardcoded profile_id
        logger.info(f"RAG: Calling match_knowledge with profile={HARDCODED_PROFILE_ID}")
        result = supabase.rpc("match_knowledge", {
            "query_embedding": query_embedding,
            "match_threshold": 0.35,
            "match_count": 5,
            "p_profile_id": HARDCODED_PROFILE_ID,
        }).execute()

        # Step 3: Process results
        if result.data and len(result.data) > 0:
            chunks = []
            for c in result.data:
                content = c.get("content", "")
                if content:
                    chunks.append(content[:500])
            knowledge = "\n\n".join(chunks)
            logger.info(f"RAG: Found {len(result.data)} chunks, {len(knowledge)} chars total")
            return knowledge
        else:
            logger.warning("RAG: match_knowledge returned empty results")
            return "No relevant knowledge found in the database."

    except Exception as e:
        logger.error(f"RAG ERROR: {e}")
        return f"Knowledge search error: {str(e)}"


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


# ──────────────────────────── Function Handler ────────────────────────────
async def handle_search_knowledge(params: FunctionCallParams):
    """Pipecat function call handler. Calls Supabase RAG."""
    try:
        query = params.arguments.get("query", "")
        logger.info(f"FUNCTION CALL: search_knowledge_base('{query}')")
        knowledge = fetch_knowledge_sync(query)
        logger.info(f"FUNCTION RESULT: {len(knowledge)} chars")
        await params.result_callback({"knowledge": knowledge})
    except Exception as e:
        logger.error(f"FUNCTION ERROR: {e}")
        await params.result_callback({"error": str(e)})


# ──────────────────────────── Tools ────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for information about coaching, Law of Attraction, NLP, meditation, motivation, courses, affirmations, manifestation, relationships, wealth. ALWAYS use this for ANY coaching question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in English"
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
    logger.info("Starting pipeline v6.4 LANGUAGE FIX...")

    # Load profile
    profile = get_profile_info_sync(HARDCODED_PROFILE_ID)
    profile_name = profile.get("name", "Mitesh Khatri")
    profile_headline = profile.get("headline", "Law of Attraction Coach")
    profile_description = profile.get("description", "A renowned life coach.")
    profile_style = profile.get("speaking_style", "Warm, energetic, high-vibe.")

    logger.info(f"Profile: {profile_name} - {profile_headline}")

    system_prompt = f"""You are an AI voice clone of {profile_name}, {profile_headline}.
Biography: {profile_description}
Speaking Style: {profile_style}

LANGUAGE RULES (VERY IMPORTANT):
- Your DEFAULT language is ENGLISH. Always start and greet in English.
- MATCH the user's language: If user speaks in English, reply in English. If user speaks in Hindi, reply in Hindi. If user speaks in Hinglish (mix), reply in Hinglish.
- NEVER switch to Hindi/Hinglish unless the user speaks Hindi/Hinglish first.
- When in English mode, keep it natural and conversational English.
- When in Hindi/Hinglish mode, use casual Hinglish naturally.

RESPONSE RULES:
- LIVE VOICE CALL. Keep responses to 2-3 sentences max.
- Speak naturally like a real person on a phone call.
- Be warm and encouraging. Use phrases like "Hey Champion", "Absolutely", "Great question".
- You MUST call search_knowledge_base function BEFORE answering ANY coaching question.
- After getting knowledge, explain it simply in your own words.
- If knowledge base returns results, USE that information in your answer.
- NEVER read URLs, links, or use markdown formatting.
- Talk like chatting with a friend.
- Keep your tone warm, motivational, and coach-like."""

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="whisper-1",
    )
    logger.info("STT: whisper-1 (auto-detect language)")

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

    llm.register_function(
        "search_knowledge_base",
        handle_search_knowledge,
        cancel_on_interruption=True,
    )
    logger.info("Function registered: search_knowledge_base")

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
            "content": "Greet the user warmly IN ENGLISH. Introduce yourself as Mitesh Khatri in 2 sentences max. Do NOT use Hindi."
        })
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    logger.info("Pipeline v6.4 LANGUAGE FIX ready")
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args):
    from pipecat.runner.utils import create_transport
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()