"""Mitesh AI Coach - Pipecat Cloud Voice Bot with RAG"""

import os
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
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TranscriptionFrame

import openai as openai_module
from supabase import create_client

load_dotenv(override=True)

logger.info("Mitesh Bot v4.0-RAG starting...")

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

# Default profile ID (will be overridden by dynamic data)
DEFAULT_PROFILE_ID = os.getenv("DEFAULT_PROFILE_ID", None)


async def fetch_knowledge(query_text: str, profile_id: str = None) -> str:
    """Fetch relevant knowledge from Supabase using vector similarity search"""
    if not supabase or not query_text.strip():
        return ""

    try:
        # Step 1: Generate embedding for the query
        embedding_response = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
        )
        query_embedding = embedding_response.data[0].embedding

        knowledge_context = ""

        # Step 2a: Try match_knowledge (chat function - with profile filter)
        if profile_id:
            try:
                result = supabase.rpc("match_knowledge", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.35,
                    "match_count": 5,
                    "p_profile_id": profile_id,
                }).execute()

                if result.data:
                    chunks = [
                        f"[{c.get('source_title', 'Source')}]: {c.get('content', '')}"
                        for c in result.data
                    ]
                    knowledge_context = "\n\n".join(chunks)
                    logger.info(f"📚 RAG: Found {len(result.data)} chunks via match_knowledge")
            except Exception as e:
                logger.warning(f"match_knowledge failed: {e}")

        # Step 2b: Fallback to match_knowledge_chunks if no results
        if not knowledge_context:
            try:
                result = supabase.rpc("match_knowledge_chunks", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.5,
                    "match_count": 3,
                }).execute()

                if result.data:
                    chunks = [c.get("content", "") for c in result.data]
                    knowledge_context = "\n\n".join(chunks)
                    logger.info(f"📚 RAG: Found {len(result.data)} chunks via match_knowledge_chunks")
            except Exception as e:
                logger.warning(f"match_knowledge_chunks failed: {e}")

        return knowledge_context

    except Exception as e:
        logger.error(f"❌ RAG Error: {e}")
        return ""


async def get_profile_info(profile_id: str) -> dict:
    """Fetch mind_profile info from Supabase"""
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


class RAGProcessor(FrameProcessor):
    """Intercepts user transcription, fetches knowledge, and enriches the LLM context"""

    def __init__(self, messages: list, profile_id: str = None):
        super().__init__()
        self.messages = messages
        self.profile_id = profile_id
        self.knowledge_injected = False

    async def process_frame(self, frame, direction):
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            user_text = frame.text.strip()
            logger.info(f"🎤 User said: {user_text}")

            # Fetch knowledge for this question
            knowledge = await fetch_knowledge(user_text, self.profile_id)

            if knowledge:
                # Update system prompt with fresh knowledge context
                knowledge_prompt = (
                    f"\n\n--- KNOWLEDGE BASE CONTEXT (USE THIS TO ANSWER - 80% WEIGHT) ---\n"
                    f"{knowledge}\n"
                    f"--- END KNOWLEDGE ---\n"
                    f"STRICT: Answer based on the above knowledge. "
                    f"If not found, say you don't have that info but offer related help."
                )

                # Update the system message with knowledge
                if len(self.messages) > 0 and self.messages[0]["role"] == "system":
                    base_prompt = self.messages[0]["content"].split("--- KNOWLEDGE BASE CONTEXT")[0]
                    self.messages[0]["content"] = base_prompt + knowledge_prompt
                    logger.info(f"📚 Knowledge injected ({len(knowledge)} chars)")

        await self.push_frame(frame, direction)


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
    logger.info("Starting Mitesh AI Coach pipeline with RAG...")

    # Extract profile_id from runner_args body (passed from frontend)
    profile_id = DEFAULT_PROFILE_ID
    if hasattr(runner_args, 'body') and runner_args.body:
        body = runner_args.body if isinstance(runner_args.body, dict) else {}
        profile_id = body.get("profile_id", body.get("profileId", DEFAULT_PROFILE_ID))

    logger.info(f"🆔 Profile ID: {profile_id}")

    # Fetch profile info
    profile = await get_profile_info(profile_id) if profile_id else {}

    # Build system prompt
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
- If knowledge base has the answer, explain it simply.
- If not, say "I don't have that info right now, but let me help with what I can!" positively.
- NEVER read out URLs or links in voice.
- NEVER use markdown formatting (no **, no ##, no bullets).
- Speak like you're talking to a friend, not writing an essay.
"""

    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
        model_id="sonic-multilingual",
    )

    messages = [{"role": "system", "content": system_prompt}]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # RAG processor - intercepts transcriptions and fetches knowledge
    rag = RAGProcessor(messages, profile_id)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rag,  # RAG processor between STT and context aggregator
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