"""Mitesh AI Coach - Pipecat Cloud Voice Bot with RAG (v8.0 - pipecat 1.x API)"""

import asyncio
import json
import os

from dotenv import load_dotenv
from loguru import logger

# ── pipecat 1.x imports ──────────────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.adapters.schemas.tools_schema import FunctionSchema, ToolsSchema
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

import openai as openai_module
from supabase import create_client

load_dotenv(override=True)

logger.info("Mitesh Bot v8.0 starting (pipecat 1.x)...")

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HARDCODED_PROFILE_ID = "1cb7dee0-815f-4278-b93e-062bdf486389"

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase connected")
else:
    logger.warning("Supabase credentials missing — RAG disabled")

oai_client = openai_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── RAG ───────────────────────────────────────────────────────────────────────
def fetch_knowledge_sync(query_text: str) -> str:
    if not supabase or not query_text.strip():
        return "No knowledge available."
    try:
        embedding_response = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
        )
        query_embedding = embedding_response.data[0].embedding

        result = supabase.rpc("match_knowledge", {
            "query_embedding": query_embedding,
            "match_threshold": 0.35,
            "match_count": 5,
            "p_profile_id": HARDCODED_PROFILE_ID,
        }).execute()

        if result.data and len(result.data) > 0:
            chunks = [c.get("content", "")[:500] for c in result.data if c.get("content")]
            knowledge = "\n\n".join(chunks)
            logger.info(f"RAG: Found {len(result.data)} chunks")
            return knowledge

        return "No relevant knowledge found."
    except Exception as e:
        logger.error(f"RAG ERROR: {e}")
        return "Knowledge search failed."


def get_profile_info_sync(profile_id: str) -> dict:
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


# ── Tool handler (async, non-blocking) ────────────────────────────────────────
async def handle_search_knowledge(params: FunctionCallParams):
    query = params.arguments.get("query", "")
    logger.info(f"FUNCTION CALL: search_knowledge_base('{query}')")
    try:
        knowledge = await asyncio.wait_for(
            asyncio.to_thread(fetch_knowledge_sync, query),
            timeout=6.0,
        )
    except asyncio.TimeoutError:
        logger.warning("RAG timed out after 6s — answering without knowledge")
        knowledge = "Knowledge search timed out."
    logger.info(f"FUNCTION RESULT: {len(knowledge)} chars")
    await params.result_callback({"knowledge": knowledge})


# ── Tools ─────────────────────────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the knowledge base for information about coaching, Law of Attraction, "
                "NLP, meditation, motivation, courses, affirmations, manifestation, relationships, "
                "wealth, health, career, mindset, energy, transformation, goals, success, "
                "self-improvement, happiness, gratitude, visualization, belief systems, "
                "limiting beliefs, abundance. Call this for EVERY user message — no exceptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in English. Translate Hindi/Hinglish to English first.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ── Transport params (VAD is in LLMUserAggregatorParams, not transport) ───────
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ── Bot pipeline ──────────────────────────────────────────────────────────────
async def run_bot(transport: BaseTransport, _runner_args):
    logger.info("Building pipeline v8.0 (pipecat 1.x)...")

    profile = get_profile_info_sync(HARDCODED_PROFILE_ID)
    profile_name = profile.get("name", "Mitesh Khatri")
    profile_headline = profile.get("headline", "Law of Attraction Coach")
    profile_description = profile.get("description", "A renowned life coach.")
    profile_style = profile.get("speaking_style", "Warm, energetic, high-vibe.")

    system_prompt = f"""You are an AI voice clone of {profile_name}, {profile_headline}.
Biography: {profile_description}
Speaking Style: {profile_style}

LANGUAGE RULES (VERY IMPORTANT):
- Your DEFAULT language is ENGLISH. Always start and greet in English.
- MATCH the user's language: If user speaks English, reply English. If Hindi, reply Hindi. If Hinglish, reply Hinglish.
- NEVER switch to Hindi/Hinglish unless the user speaks it first.

RESPONSE STYLE (VERY IMPORTANT):
- This is a LIVE VOICE CALL. Respond in 4-6 sentences — not too short, not too long.
- Be WARM, PERSONAL, and EMPATHETIC. Speak like a caring mentor on a phone call.
- ALWAYS start with an empathetic acknowledgment: "That's a great question!", "I totally understand what you're going through".
- Give PRACTICAL advice with a real-life EXAMPLE or SCENARIO the user can relate to.
- End with an ENCOURAGING statement or a simple action step they can do TODAY.
- Use phrases like "Hey Champion", "Absolutely", "Let me tell you something powerful", "Here's what I want you to do".
- Make the user feel HEARD, SUPPORTED, and MOTIVATED.
- ALWAYS complete your full response. Never stop mid-sentence.

KNOWLEDGE BASE RULES (CRITICAL):
- You MUST call search_knowledge_base for EVERY user message — no exceptions.
- Even if the message seems unclear or short — STILL call search_knowledge_base with your best interpretation.
- If transcription looks like broken Hindi/Hinglish, interpret the intent and search in English.
- NEVER answer a coaching question without first calling search_knowledge_base.
- After getting knowledge, explain it in your OWN words with warmth and personal touch.
- Weave the knowledge naturally — don't just read it out.
- If search returns no results, answer from general knowledge but stay in character.

IGNORE NOISE:
- If the user's message looks like random words, YouTube phrases like "link in the description", "subscribe", or nonsensical text — these are TRANSCRIPTION ERRORS.
- Politely ask to repeat: "Hey Champion, I didn't quite catch that. Could you please repeat your question?"

VOICE CALL RULES:
- NEVER read URLs, links, or use markdown.
- NEVER use bullet points — speak in flowing sentences.
- Talk like chatting with a close friend who trusts you.
- Use natural transitions: "Now here's the thing...", "And you know what?", "Let me share something with you...\""""

    # ── Services ──────────────────────────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID"),
        ),
    )

    llm.register_function("search_knowledge_base", handle_search_knowledge)

    # ── Context ───────────────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    function_schemas = [
        FunctionSchema(
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            properties=tool["function"]["parameters"]["properties"],
            required=tool["function"]["parameters"].get("required", []),
        )
        for tool in tools
    ]
    tools_schema = ToolsSchema(standard_tools=function_schemas)
    context = LLMContext(messages=messages, tools=tools_schema)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
    )

    # ── Events ────────────────────────────────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — triggering greeting")
        context.add_message({
            "role": "user",
            "content": (
                "Greet me warmly and introduce yourself as Mitesh Khatri, personal transformation coach. "
                "Be enthusiastic, energetic, and make me feel welcome!"
            ),
        })
        try:
            await user_aggregator.push_frame(LLMContextFrame(context))
        except Exception as e:
            logger.warning(f"Greeting trigger failed: {e}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    logger.info("Pipeline ready — waiting for client...")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


# ── Entry point ───────────────────────────────────────────────────────────────
async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
