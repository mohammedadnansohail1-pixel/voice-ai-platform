"""Streaming pipeline - LLM tokens → TTS sentences → Audio chunks."""
import asyncio
from typing import Callable, Optional, Union, Awaitable

import numpy as np

from ..core.config import Config
from ..core.types import LLMMessage
from ..logging import get_logger
from ..llm import OllamaLLM
from ..tts import KokoroTTS

logger = get_logger("engine.streaming")


class StreamingPipeline:
    """
    Streams LLM → TTS → Audio with sentence buffering.
    
    Pattern:
    1. LLM streams tokens
    2. Buffer until sentence boundary
    3. TTS synthesizes sentence
    4. Callback with audio
    5. Repeat while LLM still generating
    """
    
    def __init__(
        self,
        llm: OllamaLLM,
        tts: KokoroTTS,
        config: Config,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.config = config
        
        # State
        self.is_generating = False
        self.is_interrupted = False
    
    async def generate_streaming(
        self,
        messages: list[LLMMessage],
        on_audio: Optional[Callable[[np.ndarray, int], Union[None, Awaitable[None]]]] = None,
    ) -> tuple[str, list[np.ndarray]]:
        """
        Generate response with streaming TTS.
        
        Args:
            messages: Conversation history
            on_audio: Callback for each audio chunk (can be sync or async)
        
        Returns:
            (full_text, list of audio chunks)
        """
        self.is_generating = True
        self.is_interrupted = False
        
        audio_chunks = []
        full_response = ""
        buffer = ""
        sentence_count = 0
        
        # Sentence endings
        endings = (".", "!", "?", "。", "！", "？")
        min_sentence_len = 20
        
        async for token in self.llm.generate_stream(messages):
            if self.is_interrupted:
                break
            
            buffer += token
            full_response += token
            
            # Check for sentence boundary
            if len(buffer) >= min_sentence_len:
                for i, char in enumerate(buffer):
                    if char in endings:
                        # Check it's not a decimal or abbreviation
                        if i + 1 < len(buffer) and buffer[i + 1] not in " \n":
                            continue
                        
                        sentence = buffer[:i + 1].strip()
                        if sentence and not self.is_interrupted:
                            sentence_count += 1
                            
                            # Synthesize
                            result = self.tts.synthesize(sentence)
                            audio_chunks.append(result.audio_data)
                            
                            logger.debug(
                                "sentence_streamed",
                                num=sentence_count,
                                audio_ms=f"{result.duration_ms:.0f}",
                            )
                            
                            # Call callback (handle both sync and async)
                            if on_audio:
                                ret = on_audio(result.audio_data, result.sample_rate)
                                if asyncio.iscoroutine(ret):
                                    await ret
                        
                        buffer = buffer[i + 1:].lstrip()
                        break
        
        # Send any remaining text
        if buffer.strip() and not self.is_interrupted:
            result = self.tts.synthesize(buffer.strip())
            audio_chunks.append(result.audio_data)
            
            if on_audio:
                ret = on_audio(result.audio_data, result.sample_rate)
                if asyncio.iscoroutine(ret):
                    await ret
        
        self.is_generating = False
        return full_response, audio_chunks
    
    def interrupt(self) -> None:
        """Interrupt generation (barge-in)."""
        self.is_interrupted = True
        logger.debug("streaming_interrupted")


async def test_streaming():
    """Test streaming pipeline."""
    from ..core.config import load_config
    
    print("=" * 50)
    print("Streaming Pipeline Test")
    print("=" * 50)
    
    config = load_config("configs/base.yaml")
    
    print("\nLoading models...")
    llm = OllamaLLM(config.llm)
    tts = KokoroTTS(config.tts)
    
    pipeline = StreamingPipeline(llm, tts, config)
    
    messages = [LLMMessage(role="user", content="Tell me a short story about a robot learning to cook.")]
    
    print("\nStreaming response:")
    print("-" * 50)
    
    sentence_num = 0
    def on_audio(audio, sr):
        nonlocal sentence_num
        sentence_num += 1
        duration = len(audio) / sr
        print(f"  [Sentence {sentence_num}: {duration:.1f}s audio ready]")
    
    full_text, audio_chunks = await pipeline.generate_streaming(messages, on_audio)
    
    print("-" * 50)
    print(f"\nFull response: {full_text}")
    print(f"\nTotal sentences: {len(audio_chunks)}")
    
    total_audio_ms = sum(len(a) / 24000 * 1000 for a in audio_chunks)
    print(f"Total audio: {total_audio_ms:.0f}ms")
    
    print("\n✅ Streaming pipeline working")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    asyncio.run(test_streaming())
