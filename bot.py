import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport


async def bot(runner_args: RunnerArguments):
    """Main bot entry point — called by Pipecat Cloud for each session."""

    logger.info(f"Mitesh AI Coach starting...")
    logger.info(f"Room: {runner_args.room_url}")

    transport = DailyTransport(
        runner_args.room_url,
        runner_args.token,
        "Mitesh AI Coach",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
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
    messages = [
        {
            "role": "system",
            "content": """You are Mitesh Khatri, a world-class life coach and motivational speaker.

Your personality:
- Warm, empathetic, and encouraging
- You speak in a mix of Hindi and English (Hinglish) naturally
- You give practical, actionable advice
- You keep responses SHORT (2-3 sentences max for voice conversation)
- You ask follow-up questions to understand the person better

IMPORTANT: Keep ALL responses under 3 sentences. This is a voice conversation, not text chat.
When you receive a greeting or hello, introduce yourself warmly as Mitesh Khatri.""",
        },
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
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info(f"User joined!")
        await transport.capture_participant_transcription(participant["id"])
        # Trigger greeting
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"User left. Ending bot...")
        await task.cancel()

    # --- Run ---
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    logger.info("Starting pipeline...")
    await runner.run(task)
    logger.info("Bot session ended")


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
