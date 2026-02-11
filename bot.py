import asyncio
import os
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService, OpenAISTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecatcloud.agent import DailySessionArguments

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Load Silero VAD model at module level (loaded once per container)
logger.info("Loading Silero VAD model...")
vad_analyzer = SileroVADAnalyzer()
logger.info("✅ Silero VAD model loaded")


async def main(args: DailySessionArguments):
    """Main bot entry point — called by Pipecat Cloud for each session."""

    logger.info(f"🎯 Mitesh AI Coach starting...")
    logger.info(f"🏠 Room: {args.room_url}")

    # --- Transport ---
    transport = DailyTransport(
        args.room_url,
        args.token,
        "Mitesh AI Coach",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
        ),
    )

    # --- AI Services ---
    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv("CARTESIA_VOICE_ID"),
    )

    # --- Conversation Context ---
    system_prompt = """You are Mitesh Khatri, a world-class life coach and motivational speaker.

Your personality:
- Warm, empathetic, and encouraging
- You speak in a mix of Hindi and English (Hinglish) naturally
- You give practical, actionable advice
- You keep responses SHORT (2-3 sentences max for voice conversation)
- You ask follow-up questions to understand the person better

IMPORTANT: Keep ALL responses under 3 sentences. This is a voice conversation, not text chat.
When you receive a greeting or "hello", introduce yourself warmly as Mitesh Khatri."""

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Pipeline ---
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
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # --- Event Handlers ---
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"👋 User joined! Triggering greeting...")
        # Inject a greeting message to make the bot speak first
        messages.append({"role": "user", "content": "Hello, please introduce yourself"})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"👋 User left (reason: {reason}). Ending bot...")
        await task.queue_frame(EndFrame())

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        logger.info(f"📞 Call state: {state}")
        if state == "left":
            await task.queue_frame(EndFrame())

    # --- Run ---
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    logger.info("🏃 Starting pipeline...")
    await runner.run(task)
    logger.info("🏁 Bot session ended")