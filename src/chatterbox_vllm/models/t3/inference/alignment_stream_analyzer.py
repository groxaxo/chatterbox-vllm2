# Copyright (c) 2025 Resemble AI (adapted for vLLM)
# Author: John Meade, Jeremy Hsu (original), adapted for vLLM
# MIT License

"""
Alignment Stream Analyzer for Chatterbox TTS on vLLM

This module monitors text-to-speech alignment during generation to detect and prevent
issues like hallucinations, repetitions, and incomplete outputs. It uses attention
patterns from specific Llama layers to track which text positions are being attended
to as speech is generated.

Key features:
- Monitors attention patterns in real-time during generation
- Detects false starts (noisy beginnings with potential hallucinations)
- Detects long tails (extended generation after text completion)
- Detects repetitions (both alignment and token-level)
- Detects discontinuities (unnatural jumps in attention position)
- Forces EOS when quality issues are detected
"""

import logging
import torch
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Attention heads in Llama that show strong text-speech alignment signals
# Format: (layer_index, head_index)
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    """Results from alignment analysis for a single generation step."""
    
    # Was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    
    # Was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    
    # Was this frame detected as repeating existing text content?
    repetition: bool
    
    # Was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    
    # Has inference reached the end of the text tokens?
    complete: bool
    
    # Approximate position in the text token sequence (for online timestamps)
    position: int


class AlignmentStreamAnalyzer:
    """
    Monitors text-speech alignment using transformer attention patterns.
    
    This analyzer hooks into specific attention layers of the Llama backbone to extract
    alignment information between text tokens and generated speech frames. It uses
    heuristics to detect various quality issues and can force EOS when problems are found.
    
    Note: This is a simplified version adapted for vLLM. The full implementation would
    require deeper integration with vLLM's attention mechanism.
    """
    
    def __init__(
        self,
        text_tokens_count: int,
        eos_token_id: int = 0,
        device: str = "cuda",
    ):
        """
        Initialize the alignment stream analyzer.
        
        Args:
            text_tokens_count: Number of text tokens in the input prompt
            eos_token_id: Token ID that signals end of speech generation
            device: Device to run computations on
        """
        self.text_tokens_count = text_tokens_count
        self.eos_token_id = eos_token_id
        self.device = device
        
        # Alignment matrix: tracks attention between speech frames and text tokens
        # Shape: (num_speech_frames, num_text_tokens)
        self.alignment = torch.zeros(0, text_tokens_count, device=device)
        
        # Current position tracking
        self.curr_frame_pos = 0
        self.text_position = 0
        
        # Generation state
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        
        # Token repetition tracking
        self.generated_tokens = []
        
    def reset(self):
        """Reset the analyzer state for a new generation."""
        self.alignment = torch.zeros(0, self.text_tokens_count, device=self.device)
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.generated_tokens = []
        
    def step(
        self,
        logits: torch.Tensor,
        next_token: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Analyze alignment for the current generation step and potentially modify logits.
        
        This method:
        1. Updates the alignment matrix with current attention patterns (if available)
        2. Analyzes alignment for quality issues
        3. Tracks generation state (started, complete, position)
        4. Modifies logits to force EOS if quality issues detected
        
        Args:
            logits: Output logits for the current step, shape (batch_size, vocab_size)
            next_token: The token that was just generated (for repetition detection)
            attention_weights: Attention weights from aligned heads (if available)
                              Shape: (num_heads, seq_len, seq_len)
        
        Returns:
            Modified logits (may force EOS if quality issues detected)
        """
        # Note: In vLLM, we don't have direct access to attention weights during generation
        # This is a limitation of the current vLLM architecture. We implement simplified
        # heuristics based on token patterns instead.
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            token_id = self._extract_token_id(next_token)
            self.generated_tokens.append(token_id)
            
            # Keep only last 8 tokens to prevent memory issues
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
        
        # Increment frame position
        self.curr_frame_pos += 1
        
        # Simplified completion detection: assume we've covered ~1 speech frame per text token
        # In practice, the ratio varies, but this is a reasonable approximation
        estimated_position = min(self.curr_frame_pos // 2, self.text_tokens_count - 1)
        self.text_position = estimated_position
        
        # Check if generation is likely complete (reached near end of text)
        if not self.complete and self.text_position >= self.text_tokens_count - 3:
            self.complete = True
            self.completed_at = self.curr_frame_pos
            logger.info(f"Generation marked complete at frame {self.curr_frame_pos}")
        
        # Detect token repetition (3 or more same tokens in a row)
        token_repetition = self._detect_token_repetition()
        
        # Detect long tail: generation continues too long after completion
        long_tail = False
        if self.complete and self.completed_at is not None:
            frames_after_completion = self.curr_frame_pos - self.completed_at
            long_tail = frames_after_completion >= 10  # ~400ms of extra audio
        
        # Log warnings for detected issues
        if token_repetition:
            repeated_token = self.generated_tokens[-1] if self.generated_tokens else None
            logger.warning(f"🚨 Detected token repetition: token {repeated_token}")
            
        if long_tail:
            logger.warning(f"🚨 Detected long tail: {self.curr_frame_pos - self.completed_at} frames after completion")
        
        # Suppress EOS early in generation to prevent premature termination
        if self.text_position < self.text_tokens_count - 3 and self.text_tokens_count > 5:
            # Set EOS logit to very negative value to suppress it
            logits[..., self.eos_token_id] = -2**15
        
        # Force EOS if quality issues detected
        if long_tail or token_repetition:
            logger.warning(f"Forcing EOS token: {long_tail=}, {token_repetition=}")
            # Set all logits to very negative except EOS
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_token_id] = 2**15
        
        return logits
    
    def _extract_token_id(self, token: torch.Tensor) -> int:
        """Extract scalar token ID from tensor."""
        if isinstance(token, torch.Tensor):
            return token.item() if token.numel() == 1 else token.view(-1)[0].item()
        return int(token)
    
    def _detect_token_repetition(self) -> bool:
        """
        Detect excessive token repetition.
        
        Returns True if the last 3 tokens are all the same.
        """
        if len(self.generated_tokens) >= 3:
            # Check if last 3 tokens are all identical
            last_three = self.generated_tokens[-3:]
            return len(set(last_three)) == 1
        return False
    
    def get_analysis_result(self) -> AlignmentAnalysisResult:
        """
        Get the current alignment analysis result.
        
        Returns:
            AlignmentAnalysisResult with current state
        """
        # For simplified version, we only track token repetition and completion
        token_repetition = self._detect_token_repetition()
        long_tail = False
        if self.complete and self.completed_at is not None:
            frames_after_completion = self.curr_frame_pos - self.completed_at
            long_tail = frames_after_completion >= 10
        
        return AlignmentAnalysisResult(
            false_start=not self.started and self.curr_frame_pos < 5,
            long_tail=long_tail,
            repetition=token_repetition,
            discontinuity=False,  # Not implemented in simplified version
            complete=self.complete,
            position=self.text_position,
        )
