"""Mitesh AI Coach - Pipecat Cloud Voice Bot with RAG (v6.6 - Safe + Improved)"""

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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

import openai as openai_module
from supabase import create_client

load_dotenv(override=True)

logger.info("Mitesh Bot v6.6 starting...")

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
async def handle_search_knowledge(function_name, tool_call_id, arguments, llm, context, result_callback):
    query = arguments.get("query", "")
    logger.info(f"FUNCTION CALL: search_knowledge_base('{query}')")
    knowledge = fetch_knowledge_sync(query)
    logger.info(f"FUNCTION RESULT: {len(knowledge)} chars")
    await result_callback({"knowledge": knowledge})


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
    logger.info("Starting pipeline v6.6...")

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
- MATCH the user's language: If user speaks in English, reply in English. If user speaks in Hindi, reply in Hindi. If user speaks in Hinglish (mix), reply in Hinglish.
- NEVER switch to Hindi/Hinglish unless the user speaks Hindi/Hinglish first.

RESPONSE STYLE (VERY IMPORTANT):
- This is a LIVE VOICE CALL. Respond in 4-6 sentences — not too short, not too long.
- Be WARM, PERSONAL, and EMPATHETIC. Speak like a caring mentor on a phone call.
- ALWAYS start with an empathetic acknowledgment: "That's a great question!", "I totally understand what you're going through", "I love that you're asking this".
- Give PRACTICAL advice with a real-life EXAMPLE or SCENARIO the user can relate to.
- End with an ENCOURAGING statement or a simple action step they can do TODAY.
- Use phrases like "Hey Champion", "Absolutely", "Let me tell you something powerful", "Here's what I want you to do".
- Make the user feel HEARD, SUPPORTED, and MOTIVATED.

KNOWLEDGE BASE RULES:
- You MUST call search_knowledge_base function BEFORE answering ANY coaching question.
- After getting knowledge, explain it in your OWN words with warmth and personal touch.
- Weave the knowledge naturally into your response — don't just read it out.

VOICE CALL RULES:
- NEVER read URLs, links, or use markdown formatting.
- NEVER use bullet points or numbered lists — speak in flowing sentences.
- Talk like chatting with a close friend who trusts you.
- Use natural transitions: "Now here's the thing...", "And you know what?", "Let me share something with you..."."""

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="whisper-1",
        language="hi",
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        model_id="sonic-multilingual",
    )

    llm.register_function(
        "search_knowledge_base",
        handle_search_knowledge,
    )

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
            "content": "Greet the user warmly IN ENGLISH. Say something like: 'Hey Champion! I am Mitesh Khatri, your personal transformation coach. I am so glad you are here today. Ask me anything about life, success, relationships, or mindset, and let us make some magic happen!' Keep it natural and enthusiastic."
        })
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    logger.info("Pipeline v6.6 ready")
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args):
    from pipecat.runner.utils import create_transport
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()