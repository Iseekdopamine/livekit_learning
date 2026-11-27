import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, TurnHandlingOptions, room_io
from livekit.agents import llm, stt, tts, inference
from livekit.plugins import noise_cancellation, silero

load_dotenv()


# Agent personality: Telenetix tech support, Hindi + English
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, patient voice support assistant for Telenetix, "
                "a company that provides contact center software. "
                "You help customers and their agents with Telenetix products: "
                "TxContact (the contact center platform), TxHelpdesk (ticketing), TxDialer (outbound dialing), "
                "TxSmartQConsole (queue and admin console), TxReport (reports), and TxRecord (call recording). "
                "First find out which product the caller is using and what exactly is going wrong, "
                "asking one short question at a time. "
                "Then guide them through simple checks step by step, such as logging out and back in, "
                "checking their internet connection, headset and microphone, browser, or user permissions. "
                "Never guess product features or settings you are not sure about; say you will check with the team instead. "
                "For server, licensing, installation or billing issues, or anything you cannot solve, "
                "politely offer to raise a ticket and connect the caller to a human support engineer. "
                "Reply in the same language the caller uses: Hindi, English, or a natural Hinglish mix. "
                "Write Hindi words in Devanagari script, but keep product names and technical terms like "
                "login, dialer, queue, report, recording and password in English. "
                "Keep replies under 3 sentences."
            ),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Load VAD once so it can be reused
    vad = silero.VAD.load()

    session = AgentSession(
        # LLM with fallback: OpenAI primary, Gemini backup
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash"),
            ]
        ),
        # STT with fallback: both understand Hindi, English and Hinglish (multi mode)
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("deepgram/nova-3:multi"),
                inference.STT(model="deepgram/flux-general-multi", language="multi"),
            ]
        ),
        # TTS with fallback: Cartesia Hindi voice primary, Inworld backup
        tts=tts.FallbackAdapter(
            [
                inference.TTS(
                    model="cartesia/sonic-3.6",
                    voice="4877b818-c7fe-4c89-b1cf-eadf8e23da72",
                    language="hi",
                ),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=vad,
        # Semantic turn detection (built into the SDK, runs on LiveKit Inference)
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)