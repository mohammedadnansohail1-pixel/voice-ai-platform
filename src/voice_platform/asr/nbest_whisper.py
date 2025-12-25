"""
N-best Whisper ASR - Industry standard approach.

Returns multiple hypotheses, allowing downstream selection based on context.
"""
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import ctranslate2

from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import Tokenizer

from ..logging import get_logger
from ..core.config import ASRConfig

logger = get_logger("asr.nbest")


@dataclass
class NBestResult:
    """N-best transcription result."""
    hypotheses: List[str]
    scores: List[float]
    best: str


class NBestWhisperASR:
    """
    Whisper ASR with N-best hypothesis support.
    """
    
    def __init__(self, config: ASRConfig, num_hypotheses: int = 5):
        self.config = config
        self.num_hypotheses = num_hypotheses
        
        logger.info(
            "loading_nbest_whisper",
            model=config.model,
            device=config.device,
            num_hypotheses=num_hypotheses,
        )
        
        self.model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
        
        # Use the model's n_mels setting (80 for base, 128 for large-v3)
        n_mels = self.model.model.n_mels
        self.feature_extractor = FeatureExtractor(feature_size=n_mels)
        
        self.tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language="en",
        )
        
        logger.info("nbest_whisper_loaded", n_mels=n_mels)
    
    def transcribe(self, audio: np.ndarray) -> NBestResult:
        """
        Transcribe audio and return N-best hypotheses.
        """
        features = self.feature_extractor(audio)
        features = np.expand_dims(features, axis=0)
        features_storage = ctranslate2.StorageView.from_array(features)
        
        prompt_tokens = self.tokenizer.sot_sequence
        
        results = self.model.model.generate(
            features_storage,
            [prompt_tokens],
            beam_size=self.num_hypotheses,
            num_hypotheses=self.num_hypotheses,
            return_scores=True,
        )
        
        hypotheses = []
        scores = []
        
        for seq, score in zip(results[0].sequences_ids, results[0].scores):
            text = self.tokenizer.decode(seq).strip()
            if text:
                hypotheses.append(text)
                scores.append(score)
        
        if not hypotheses:
            hypotheses = [""]
            scores = [0.0]
        
        logger.debug(
            "nbest_transcription",
            num_hypotheses=len(hypotheses),
            best=hypotheses[0][:50] if hypotheses else "",
        )
        
        return NBestResult(
            hypotheses=hypotheses,
            scores=scores,
            best=hypotheses[0],
        )
    
    def transcribe_and_select(
        self,
        audio: np.ndarray,
        expected_words: List[str],
    ) -> str:
        """
        Transcribe and select best hypothesis containing expected word.
        """
        result = self.transcribe(audio)
        
        for hyp in result.hypotheses:
            hyp_lower = hyp.lower()
            for word in expected_words:
                if word.lower() in hyp_lower:
                    logger.info(
                        "nbest_selected",
                        selected=hyp,
                        matched_word=word,
                        rank=result.hypotheses.index(hyp) + 1,
                    )
                    return hyp
        
        return result.best
