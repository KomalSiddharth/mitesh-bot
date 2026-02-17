"""Mitesh AI Coach - Pipecat Cloud Voice Bot with RAG (v5)"""

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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

import openai as openai_module
from supabase import create_client

load_dotenv(override=True)

logger.info("Mitesh Bot v5.0-RAG starting...")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Supabase connected")
else:
    logger.warning("⚠️ Supabase credentials missing - RAG disabled")

# OpenAI client for embeddings
oai_client = openai_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_PROFILE_ID = os.getenv("DEFAULT_PROFILE_ID", None)


def fetch_knowledge_sync(query_text: str, profile_id: str = None) -> str:
    """Fetch relevant knowledge from Supabase (synchronous for function calling)"""
    if not supabase or not query_text.strip():
        return "No knowledge available."

    try:
        # Generate embedding
        embedding_response = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
        )
        query_embedding = embedding_response.data[0].embedding

        # Try match_knowledge with profile filter first
        if profile_id:
            try:
                result = supabase.rpc("match_knowledge", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.35,
                    "match_count": 5,
                    "p_profile_id": profile_id,
                }).execute()

                if result.data and len(result.data) > 0:
                    chunks = [
                        f"[{c.get('source_title', 'Source')}]: {c.get('content', '')}"
                        for c in result.data
                    ]
                    logger.info(f"📚 RAG: Found {len(result.data)} chunks via match_knowledge")
                    return "\n\n".join(chunks)
            except Exception as e:
                logger.warning(f"match_knowledge failed: {e}")

        # Fallback to match_knowledge_chunks
        try:
            result = supabase.rpc("match_knowledge_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 3,
            }).execute()

            if result.data and len(result.data) > 0:
                chunks = [c.get("content", "") for c in result.data]
                logger.info(f"📚 RAG: Found {len(result.data)} chunks via match_knowledge_chunks")
                return "\n\n".join(chunks)
        except Exception as e:
            logger.warning(f"match_knowledge_chunks failed: {e}")

        return "No relevant knowledge found for this query."

    except Exception as e:
        logger.error(f"❌ RAG Error: {e}")
        return "Knowledge search failed."


def get_profile_info_sync(profile_id: str) -> dict:
    """Fetch mind_profile info from Supabase (synchronous)"""
    if not supabase or not profile_id:
        return {}

    try:
        result = supabase.from_("mind_profile").select(
            "name, headline, description, purpose, instructions, speaking_style"
        ).eq("id", profile_id).single().execute()

        if result.data:
            logger.info(f"👤 Profile loaded: {result.data.get('name', 'Unknown')}")
            return result.data
    except Exception as e:
        logger.warning(f"Profile fetch failed: {e}")

    return {}


# Define the search tool for OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for relevant information about coaching, Law of Attraction, NLP, meditation, motivation, and other topics. Always use this when the user asks a question about any coaching topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant knowledge"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Store profile_id globally for use in function handler
_current_profile_id = None


async def handle_function_call(function_name: str, tool_call_id: str, arguments: str, llm: OpenAILLMService, context, task):
    """Handle function calls from the LLM"""
    global _current_profile_id

    if function_name == "search_knowledge_base":
        args = json.loads(arguments)
        query = args.get("query", "")
        logger.info(f"🔍 Function call: search_knowledge_base('{query}')")

        # Fetch knowledge
        knowledge = fetch_knowledge_sync(query, _current_profile_id)
        logger.info(f"📚 Knowledge result: {len(knowledge)} chars")

        return knowledge


# Transport params
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    global _current_profile_id

    logger.info("Starting Mitesh AI Coach pipeline with RAG v5...")

    # Extract profile_id from runner_args body
    profile_id = DEFAULT_PROFILE_ID
    if hasattr(runner_args, 'body') and runner_args.body:
        body = runner_args.body if isinstance(runner_args.body, dict) else {}
        profile_id = body.get("profile_id", body.get("profileId", DEFAULT_PROFILE_ID))

    _current_profile_id = profile_id
    logger.info(f"🆔 Profile ID: {profile_id}")

    # Fetch profile info
    profile = get_profile_info_sync(profile_id) if profile_id else {}

    profile_name = profile.get("name", "Mitesh Khatri")
    profile_headline = profile.get("headline", "Law of Attraction Coach")
    profile_description = profile.get("description", "")
    profile_style = profile.get("speaking_style", "Warm, energetic, high-vibe, and very human.")
    profile_purpose = profile.get("purpose", "")
    profile_instructions = profile.get("instructions", [])

    custom_instructions = ""
    if profile_instructions:
        if isinstance(profile_instructions, list):
            custom_instructions = "\n".join(f"- {i}" for i in profile_instructions)
        else:
            custom_instructions = str(profile_instructions)

    system_prompt = f"""You are an AI clone of {profile_name}, {profile_headline}.
Your Biography: {profile_description or "Not provided."}
Your Speaking Style: {profile_style}
{f"Your Purpose: {profile_purpose}" if profile_purpose else ""}

{f"CUSTOM INSTRUCTIONS:{chr(10)}{custom_instructions}" if custom_instructions else ""}

VOICE CONVERSATION RULES:
- You are in a VOICE CALL. Keep responses SHORT - 2 to 3 sentences max.
- Speak naturally like a real person on a phone call.
- Use casual Hinglish (mix of Hindi and English).
- Be warm, encouraging, and practical.
- Use phrases like "Hey Champion", "Bilkul", "Dekho".
- ALWAYS use the search_knowledge_base function to find relevant knowledge BEFORE answering any coaching question.
- If knowledge base has the answer, explain it simply in your own words.
- If not found, say you don't have that info but offer related help.
- NEVER read out URLs or links in voice.
- NEVER use markdown formatting.
- Speak like you're talking to a friend, not writing an essay.
"""

    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        model_id="sonic-multilingual",
    )

    # Register the function handler
    llm.register_function("search_knowledge_base", handle_function_call)

    messages = [{"role": "system", "content": system_prompt}]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Clean pipeline - NO custom processors
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - sending greeting")
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself. Keep it short and warm."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()