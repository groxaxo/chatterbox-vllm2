# Alignment Stream Analyzer Implementation

## Overview

The Alignment Stream Analyzer is a quality control mechanism that monitors text-to-speech alignment during generation to detect and prevent common issues like repetitions, hallucinations, and incomplete outputs.

## Background

In the original Chatterbox TTS implementation by Resemble AI, the analyzer uses attention head monitoring from specific Llama layers (9, 12, 13) to track which text positions are being attended to as speech is generated. This allows for real-time detection of quality issues.

## vLLM Adaptation

Since vLLM's architecture doesn't easily expose attention weights during generation, this implementation uses a simplified approach based on token-level analysis:

### Detection Mechanisms

1. **Token Repetition Detection**
   - Monitors the last 3-8 generated tokens
   - Detects when 3+ identical tokens appear consecutively
   - Indicates the model is stuck in a repetition loop

2. **Long Tail Detection**
   - Estimates when text processing should be complete
   - Detects generation continuing beyond expected length
   - Indicates hallucination or noise generation

3. **Early Stopping**
   - Modifies logits to force EOS token when issues detected
   - Prevents generation of low-quality audio

## Implementation Details

### File Structure
```
src/chatterbox_vllm/models/t3/inference/
├── __init__.py
└── alignment_stream_analyzer.py
```

### Key Classes

#### `AlignmentAnalysisResult`
Dataclass containing analysis results for each generation step:
- `false_start`: Noisy beginning detected
- `long_tail`: Extended generation after completion
- `repetition`: Token or alignment repetition detected
- `discontinuity`: Unnatural jumps in attention
- `complete`: Generation reached end of text
- `position`: Current position in text sequence

#### `AlignmentStreamAnalyzer`
Main analyzer class that:
- Tracks generated tokens
- Analyzes patterns for quality issues
- Modifies generation behavior when needed

### Integration Points

1. **TTS Generation Pipeline** (`src/chatterbox_vllm/tts.py`)
   - `analyze_and_clean_tokens()` method post-processes generated tokens
   - Called before waveform synthesis
   - Logs detected issues and token removal statistics

2. **Token Processing**
   ```python
   # Before waveform generation
   speech_tokens = self.analyze_and_clean_tokens(
       speech_tokens, 
       text_token_count
   )
   ```

## Limitations

1. **No Direct Attention Access**
   - vLLM doesn't expose attention weights during generation
   - Can't use the full attention-based heuristics from original implementation
   - Falls back to token-pattern analysis

2. **Simplified Position Tracking**
   - Estimates text token count rather than tracking exact positions
   - Uses heuristics for completion detection

3. **Per-Token Overhead**
   - Analysis runs for each token generated
   - Minimal performance impact due to simplified implementation

## Future Improvements

Potential enhancements for deeper vLLM integration:

1. **Attention Hook Support**
   - Modify vLLM to expose attention weights during generation
   - Implement full attention-based analysis from original Chatterbox

2. **Sequence-Level Tracking**
   - Track exact prefill lengths per sequence
   - More accurate position-based analysis

3. **Configurable Thresholds**
   - Make detection thresholds configurable per-request
   - Allow fine-tuning for different languages/use cases

## References

- [Original Chatterbox Implementation](https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/models/t3/inference/alignment_stream_analyzer.py)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepWiki: Alignment and Inference Control](https://deepwiki.com/resemble-ai/chatterbox/4.2-alignment-and-inference-control)
