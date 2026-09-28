  # Local Speech-Interaction Pipeline — Meeting Notes

  Date: 24 August 2026

  ## 1. Objective

  The objective was to reproduce, extend and evaluate a speech-interaction pipeline using a
  ReSpeaker microphone.

  Two language-model configurations were implemented:

  ### Offline pipeline

  User speech
  → ReSpeaker
  → Faster Whisper
  → local Qwen
  → Piper
  → speaker

  ### Cloud pipeline

  User speech
  → ReSpeaker
  → Faster Whisper
  → Gemini API
  → Piper
  → speaker

  Neither implementation uses the OpenAI API.

  ## 2. Hardware and software

  ### Laptop

  - Operating system: Ubuntu 24.04
  - ROS version: ROS 2 Jazzy
  - CPU: Intel Core i5-1135G7
  - RAM: approximately 8 GB
  - Integrated GPU: Intel Iris Xe
  - Dedicated GPU: NVIDIA GeForce MX330
  - Microphone: ReSpeaker microphone array
  - Audio output: external/laptop speaker

  ### Speech recognition

  - Engine: Faster Whisper
  - Model: base
  - Execution device: CPU
  - Compute type: int8

  ### Offline language model

  - Runtime: Ollama
  - Ollama version: 0.32.15
  - Model: qwen2.5:1.5b
  - Execution mode: CPU AVX2
  - Model keep-alive: 10 minutes

  ### Cloud language model

  - Provider: Google Gemini API
  - Model: gemini-3.5-flash-lite
  - API: REST generateContent
  - Authentication: GEMINI_API_KEY environment variable
  - Maximum output per response: 60 tokens
  - Internet connection required
  - Google Search grounding not enabled
  - Free-tier API used for testing
  - The project dashboard displayed a limit of 250,000 tokens per minute; separate
    requests-per-minute and requests-per-day limits also apply.

  ### Text-to-speech

  - Engine: Piper
  - Piper version: 1.7.0
  - Voice: en_US-lessac-medium
  - Output format: WAV
  - Sample rate: 22,050 Hz
  - Output channels: mono
  - Sample format: signed 16-bit audio

  ## 3. Audio configuration

  The ReSpeaker is recorded using:

  ALSA device: hw:1,0
  Sample rate: 16,000 Hz
  Channels: 6
  Sample width: 16-bit
  Format: signed little-endian PCM

  All six ReSpeaker channels are captured. Channel 0 is currently selected and passed to
  Faster Whisper.

  The speech endpoint rules are:

  - Wait up to 15 seconds for speech to begin.
  - Stop a turn after approximately 0.8 seconds of silence.
  - Maximum utterance duration: 20 seconds.
  - Preserve approximately 0.5 seconds of pre-roll audio.
  - If nobody speaks for 15 seconds, start listening again.

  The complete conversation has no fixed duration or turn limit. Only each individual
  utterance is limited to 20 seconds.

  ## 4. Working program files

   File                               Purpose
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   respeaker_vad_direction_test.py    ReSpeaker VAD and sound-direction test
  ─────────────────────────────────  ────────────────────────────────────────
   live_stt_respeaker.py              ReSpeaker → Whisper → terminal text
  ─────────────────────────────────  ────────────────────────────────────────
   speech_echo_respeaker.py           ReSpeaker → Whisper → Piper echo
  ─────────────────────────────────  ────────────────────────────────────────
   speech_qwen_respeaker.py           Continuous offline Qwen conversation
  ─────────────────────────────────  ────────────────────────────────────────
   speech_gemini_respeaker.py         Continuous online Gemini conversation

  ## 5. Development stages completed

  ### Stage 1: Live speech recognition

  User speech
  → ReSpeaker
  → Faster Whisper
  → recognized text

  Status: completed.

  ### Stage 2: Piper test

  Typed text
  → Piper
  → WAV file
  → speaker

  Status: completed.

  ### Stage 3: Speech echo

  User speech
  → ReSpeaker
  → Whisper
  → Piper speaks the transcription

  Status: completed.

  ### Stage 4: Local language model

  Text prompt
  → Ollama
  → Qwen2.5 1.5B
  → text response

  Status: completed.

  ### Stage 5: Offline speech-to-speech pipeline

  ReSpeaker
  → Whisper
  → Qwen
  → Piper
  → speaker

  Status: completed.

  ### Stage 6: Continuous interaction

  Listen
  → transcribe
  → generate response
  → speak response
  → listen again

  Status: completed.

  ### Stage 7: Gemini cloud model

  Text prompt
  → Gemini 3.5 Flash-Lite
  → text response

  Status: completed.

  ### Stage 8: Gemini speech-to-speech pipeline

  ReSpeaker
  → Whisper
  → Gemini
  → Piper
  → speaker

  Status: completed.

  ## 6. Latency definitions

  ### Utterance duration

  Time between the detected beginning and detected end of the user’s speech.

  ### Whisper inference time

  Time Faster Whisper takes to process the recorded audio.

  ### Speech-start-to-text time

  Time from the detected beginning of speech until the final transcription is available.

  ### Speech-end-to-text latency

  Time from the last detected speech until the final transcription is available.

  This includes:

  end-of-speech detection
  + Whisper inference
  + small program overhead

  ### Language-model response time

  Time Qwen or Gemini takes to produce the complete response text.

  ### Piper synthesis time

  Time Piper takes to convert the response text into a WAV file.

  ### Text-to-speaker-start time

  Time from completion of the Whisper transcription until speaker playback begins.

  This includes:

  language-model response
  + Piper synthesis
  + small program overhead

  ### Speech-end-to-speaker-start latency

  Time from the user finishing their speech until the robot begins speaking.

  User finishes speaking
  → endpoint detection
  → Whisper
  → Qwen or Gemini
  → Piper
  → speaker begins

  This is the primary conversational response-latency measurement.

  ### Playback duration

  Time required to play the complete generated answer.

  Playback duration depends on response length and is therefore not the primary processing-
  latency metric.

  ## 7. STT-to-Piper echo baseline

  Three short-sentence tests were collected before adding a language model.

   Measurement                    Run 1     Run 2     Run 3    Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
   Utterance duration            3.38 s    3.25 s    3.87 s     3.50 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Whisper inference             0.45 s    0.49 s    0.46 s     0.47 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Speech start → text           4.68 s    4.72 s    5.19 s     4.86 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Speech end → text             1.30 s    1.47 s    1.31 s     1.36 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Piper synthesis               2.02 s    1.26 s    1.31 s     1.53 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Speech end → speaker start    3.33 s    2.73 s    2.62 s     2.89 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Playback duration             3.96 s    3.56 s    3.93 s     3.82 s

  ### Echo-test observations

  - Two runs reproduced the reference sentence correctly.
  - One run omitted the word “the.”
  - No utterance was cut off.
  - A longer utterance of approximately 16.37 seconds was transcribed successfully.
  - Piper successfully spoke the Whisper transcription.

  ## 8. Initial Whisper-Qwen-Piper tests

  ### Cold or partly loaded Qwen test

  Transcription:

  Hello Robot. Hello Robot, what is your name?

  Qwen response:

  Hello! I'm Qwen, created by Alibaba Cloud.

   Measurement                    Result
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━
   Utterance duration             2.25 s
  ────────────────────────────  ─────────
   Whisper inference              0.46 s
  ────────────────────────────  ─────────
   Speech end → text              1.33 s
  ────────────────────────────  ─────────
   Qwen response                  7.45 s
  ────────────────────────────  ─────────
   Piper synthesis                4.12 s
  ────────────────────────────  ─────────
   Speech end → speaker start    12.91 s
  ────────────────────────────  ─────────
   Playback duration              4.22 s

  ### Warm Qwen test

  Transcription:

  What is the capital of Germany?

  Qwen response:

  The capital of Germany is Berlin.

   Measurement                   Result
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━
   Utterance duration            1.63 s
  ────────────────────────────  ────────
   Whisper inference             0.43 s
  ────────────────────────────  ────────
   Speech end → text             1.43 s
  ────────────────────────────  ────────
   Qwen response                 0.71 s
  ────────────────────────────  ────────
   Piper synthesis               2.01 s
  ────────────────────────────  ────────
   Speech end → speaker start    4.16 s
  ────────────────────────────  ────────
   Playback duration             1.93 s

  These results demonstrated the difference between cold and warm Qwen operation.

  ## 9. Qwen cold versus warm behaviour

  A cold model is not loaded in memory. Ollama must load it from disk into RAM before
  generating a response.

  A warm model is already loaded and can immediately process a request.

   Condition                              Qwen response    Speech end → speaker start
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Cold or partly loaded                         7.45 s                       12.91 s
  ──────────────────────────────  ──────────────────────  ────────────────────────────
   Warm example                                  0.71 s                        4.16 s
  ──────────────────────────────  ──────────────────────  ────────────────────────────
   Warm after automatic warm-up    approximately 0.53 s          approximately 3.07 s

  The program warms Qwen before displaying:

  SPEAK NOW

  This moves the cold-loading delay to application startup instead of making the user wait
  after asking the first question.

  ## 10. Automatic Qwen warm-up tests

  Two tests were performed after adding automatic Qwen warm-up.

   Measurement                   Test 1    Test 2    Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
   Utterance duration            1.88 s    2.75 s     2.32 s
  ────────────────────────────  ────────  ────────  ─────────
   Whisper inference             0.42 s    0.45 s     0.44 s
  ────────────────────────────  ────────  ────────  ─────────
   Speech start → text           3.15 s    4.18 s     3.67 s
  ────────────────────────────  ────────  ────────  ─────────
   Speech end → text             1.28 s    1.43 s     1.36 s
  ────────────────────────────  ────────  ────────  ─────────
   Qwen response                 0.52 s    0.54 s     0.53 s
  ────────────────────────────  ────────  ────────  ─────────
   Piper synthesis               1.18 s    1.19 s     1.19 s
  ────────────────────────────  ────────  ────────  ─────────
   Text → speaker start          1.71 s    1.74 s     1.73 s
  ────────────────────────────  ────────  ────────  ─────────
   Speech end → speaker start    2.98 s    3.16 s     3.07 s
  ────────────────────────────  ────────  ────────  ─────────
   Playback duration             2.31 s    2.57 s     2.44 s

  Primary result before continuous mode:

  > The robot began answering approximately 3.07 seconds after the user finished speaking.

  The questions and generated responses were different, so this was an operational baseline
  rather than a strict controlled comparison.

  ## 11. Ollama CPU versus Vulkan comparison

  Ollama initially used a mixed CPU/GPU Vulkan configuration.

  The Ollama log showed:

  26 of 29 model layers offloaded to Vulkan
  approximately 852 MB in Vulkan model memory
  approximately 265 MB in host memory

  ### Mixed Vulkan result

  Total duration: 25.65 seconds
  Prompt processing: 2.27 seconds
  Generation: 23.38 seconds
  Generated tokens: 14
  Generation speed: approximately 0.60 tokens/second

  ### CPU AVX2 cold result

  Total duration: 28.22 seconds
  Model loading: 27.26 seconds
  Generation: 0.61 seconds
  Generated tokens: 14

  ### CPU AVX2 warm result

  Total duration: 0.72 seconds
  Generation speed: approximately 22.65 tokens/second

  ### Comparison

   Configuration                       Generation speed    Controlled request total
  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
   Mixed Vulkan CPU/GPU     approximately 0.60 tokens/s                     25.65 s
  ──────────────────────  ──────────────────────────────  ──────────────────────────
   CPU AVX2, warm          approximately 22.65 tokens/s                      0.72 s

  CPU AVX2 generated tokens approximately 38 times faster than the Vulkan configuration on
  this laptop.

  Ollama is therefore configured to use CPU AVX2:

  OLLAMA_LLM_LIBRARY=cpu_avx2
  CUDA_VISIBLE_DEVICES=-1
  GGML_VK_VISIBLE_DEVICES=-1

  ## 12. Continuous-conversation implementation

  The program supports repeated interactions without restarting.

  Start Whisper once
  → prepare language model
  → display SPEAK NOW
  → record one user turn
  → stop microphone
  → transcribe speech
  → generate response
  → synthesize speech
  → play response
  → display SPEAK NOW again
  → repeat

  The microphone is stopped while Piper plays. This prevents the system from recording and
  responding to its own voice.

  The conversation can be stopped using:

  - “Goodbye”
  - “Good bye”
  - “Stop conversation”
  - “Exit”
  - “Quit”
  - Ctrl+C

  Both goodbye and good bye are supported because Whisper may produce either spelling.

  ## 13. Continuous Qwen test

  The first continuous Qwen test used four question-and-answer turns followed by a spoken
  exit command.

  The initial cold Qwen warm-up took:

  25.94 seconds

  This delay occurred before interaction began.

   Measurement                   Turn 1    Turn 2    Turn 3    Turn 4    Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
   Utterance duration            1.50 s    2.38 s    3.75 s    1.75 s     2.35 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Whisper inference             0.43 s    0.47 s    0.46 s    0.43 s     0.45 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Speech start → text           2.78 s    3.82 s    5.07 s    3.03 s     3.68 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Speech end → text             1.28 s    1.45 s    1.32 s    1.28 s     1.33 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Qwen response                 0.45 s    0.64 s    0.46 s    0.67 s     0.56 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Piper synthesis               2.37 s    1.20 s    1.18 s    1.20 s     1.49 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Text → speaker start          2.82 s    1.84 s    1.64 s    1.87 s     2.04 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Speech end → speaker start    4.10 s    3.29 s    2.96 s    3.15 s     3.38 s
  ────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Playback duration             2.06 s    2.32 s    1.80 s    2.82 s     2.25 s

  The exit-command turn was excluded from these averages.

  ### Main continuous Qwen result

  - Average warm Qwen response: 0.56 seconds
  - Average speech-end-to-speaker-start latency: 3.38 seconds

  ## 14. Qwen interaction observations

  ### Successful interaction

  Transcription:

  What is the capital of Germany?

  Response:

  The capital of Germany is Berlin.

  ### STT error propagation

  Intended question:

  How many states does Germany have?

  Whisper transcription:

  How many stairs does Germany have?

  Qwen response:

  Germany has 14,000 stairs.

  The error propagated through the entire pipeline:

  STT error
  → incorrect LLM interpretation
  → incorrect spoken response

  When the question was repeated clearly, Whisper detected “states” and Qwen returned the
  intended answer.

  ### Spoken exit

  Whisper sometimes transcribed “Goodbye” as:

  Good bye

  The exit list was updated to support both forms. The spoken exit then worked correctly.

  ## 15. Gemini API test

  Gemini was tested independently before integration.

  Test question:

  What is the capital of Germany?

  Response:

  The capital of Germany is Berlin.

  API metadata:

  Model: gemini-3.5-flash-lite
  Prompt tokens: 13
  Output tokens: 7
  Total tokens: 20
  Finish reason: STOP

  This confirmed:

  - API key validity
  - Network access
  - Model availability
  - Free-tier project access
  - Correct REST request structure

  The API key is not stored in the source file. It is supplied using:

  GEMINI_API_KEY

  ## 16. Continuous Gemini test

  Seven question-and-answer turns were completed, followed by a successful spoken exit.

   Measurement                   Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━
   Utterance duration             1.66 s
  ────────────────────────────  ─────────
   Whisper inference              0.42 s
  ────────────────────────────  ─────────
   Speech start → text            2.99 s
  ────────────────────────────  ─────────
   Speech end → text              1.33 s
  ────────────────────────────  ─────────
   Gemini response                0.69 s
  ────────────────────────────  ─────────
   Piper synthesis                1.80 s
  ────────────────────────────  ─────────
   Text → speaker start           2.49 s
  ────────────────────────────  ─────────
   Speech end → speaker start     3.82 s
  ────────────────────────────  ─────────
   Playback duration              2.92 s

  The “Goodbye” exit turn was excluded from these averages.

  ### Main continuous Gemini result

  - Average Gemini response: 0.69 seconds
  - Average speech-end-to-speaker-start latency: 3.82 seconds
  - No local model-loading warm-up was required.

  ## 17. Gemini interaction observations

  ### Greeting

  Transcription:

  Hello

  Gemini response:

  Hello! How can I help you today?

  ### Capital question

  Transcription:

  What is the capital of Germany?

  Gemini response:

  The capital of Germany is Berlin.

  ### Incomplete STT input

  Whisper transcription:

  States does Germany have.

  Gemini inferred the intended meaning and responded:

  Germany has sixteen federal states.

  ### Chief Minister misunderstanding

  Transcription:

  Who is the Chief Minister of India?

  Gemini correctly explained that India does not have one national Chief Minister and
  distinguished that role from the Prime Minister.

  ### Spoken exit

  Transcription:

  Goodbye

  Program output:

  Exit command detected.
  Conversation ended.

  ## 18. Time-sensitive information limitation

  Gemini is accessed through the internet, but the standard model request does not
  automatically perform a live Google Search.

  Google Search grounding was not enabled in this implementation.

  Consequently, Gemini returned some outdated political information:

  - It identified M. K. Stalin as the current Chief Minister of Tamil Nadu. Current
    official Tamil Nadu sources identify C. Joseph Vijay.

  - It identified Pinarayi Vijayan as the current Chief Minister of Kerala. Kerala’s
    official General Administration Department currently lists V. D. Satheesan.

  - Its response identifying Droupadi Murmu as President of India matched the official
    President’s Secretariat.

  Official references:

  - Tamil Nadu Lok Bhavan
    (https://lokbhavan.tn.gov.in/gallery/the-newly-sworn-in-cabinet-headed-by-thiru-c-josep
h-vijay-honble-cm-of-tamil-nadu-with-thiru-rajendra-vishwanath-arlekar-honble-governor-of-t
amil-nadu-at-lok-bhavan-chennai-today-21-05-2026/)

  - Kerala General Administration Department (https://gad.kerala.gov.in/en/chief-ministers)
  - President of India (https://www.presidentofindia.gov.in/about-presidents-secretariat)

  Internet connectivity alone must not be treated as proof that an answer is current.

  For demonstrations without search grounding:

  - Avoid questions containing “current,” “today,” “latest” or “now.”
  - Alternatively, instruct the model to state that it cannot verify current information.

  ## 19. Preliminary Qwen versus Gemini comparison

   Property                       Qwen                        Gemini
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Model                          Qwen2.5 1.5B                Gemini 3.5 Flash-Lite
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Location                       Local laptop                Google cloud
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Internet required              No                          Yes
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   API key required               No                          Yes
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Local warm-up required         Yes                         No
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Cold startup                   approximately 16–26 s       no local model loading
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Average model response         0.56 s warm                 0.69 s
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Average speech end →           3.38 s                      3.82 s
   speaker start
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Current information            Internal model knowledge    Internal knowledge unless
                                                              grounding is enabled
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Offline operation              Yes                         No
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Quota                          No API quota                Free-tier API limits
  ─────────────────────────────  ──────────────────────────  ──────────────────────────────
   Privacy                        Text remains local          Transcribed text is sent to
                                                              Google

  The questions and generated answer lengths were different, so this is a preliminary
  operational comparison rather than a strict controlled benchmark.

  ### Interpretation

  Qwen advantages:

  - Fully offline
  - No API quota
  - No cloud dependency
  - Slightly faster warm response in these tests
  - User text remains local

  Qwen disadvantages:

  - Large cold-start delay
  - Uses laptop RAM and CPU
  - Lower response quality in some cases
  - No current-information source

  Gemini advantages:

  - No local model loading
  - No local LLM memory requirement
  - Stronger handling of some incomplete or ambiguous input
  - Easy access through a REST API

  Gemini disadvantages:

  - Requires internet
  - Requires an API key
  - Subject to free-tier quotas
  - Sends transcribed user text to Google
  - Does not automatically provide live information
  - Can still return outdated or incorrect answers

  ## 20. Accuracy and reliability findings

  ### Successful behaviour

  - Six-channel ReSpeaker recording works.
  - Faster Whisper provides usable English transcriptions.
  - Both Qwen and Gemini generate short responses.
  - Piper produces intelligible speech.
  - Continuous interaction works without restarting.
  - The microphone is stopped during Piper playback.
  - The robot does not respond to its own generated speech.
  - Spoken exit commands work.
  - Qwen works offline.
  - Gemini works through the cloud API.
  - Both pipelines measure per-turn latency.

  ### Main risks

  #### STT error propagation

  User speech
  → incorrect Whisper transcription
  → incorrect model response
  → incorrect spoken answer

  #### Language-model hallucination

  A language model may confidently answer an unclear or incorrect transcription.

  #### Outdated information

  Both Qwen and ungrounded Gemini may provide outdated answers about current events or
  public officials.

  #### Network dependency

  Gemini cannot operate when the internet or Gemini API is unavailable.

  ## 21. Current limitations

  - Conversation history is not preserved.
  - Each question is processed independently.
  - STT confidence is not checked.
  - The system does not ask for clarification after uncertain transcription.
  - Only ReSpeaker channel 0 is passed to Whisper.
  - Background-noise calibration runs for one second before every turn.
  - A 0.8-second pause ends the current turn.
  - Each utterance is limited to 20 seconds.
  - Piper generates a complete WAV file before playback starts.
  - Piper output is not streamed.
  - Qwen cold startup can take approximately 16–26 seconds.
  - Gemini requires internet connectivity.
  - Gemini requires a valid API key.
  - Gemini free-tier quotas apply.
  - Gemini Search grounding is not enabled.
  - Camera and eye-contact activation are not integrated.
  - Atul’s camera/eye-contact repository is unavailable.
  - Social Marketplace architecture has been inspected but is currently parked.
  - Unitree integration is intentionally reserved for a later stage.

  ## 22. Work completed

  - Verified Ubuntu and ROS installation
  - Inspected llm_based_speech_interaction
  - Identified the available STT implementations
  - Selected Faster Whisper
  - Tested saved-audio transcription
  - Tested laptop microphone transcription
  - Tested ReSpeaker recording
  - Tested six-channel ReSpeaker audio
  - Tested voice activity and sound direction
  - Implemented speech and silence detection
  - Implemented pre-roll capture
  - Measured Whisper latency
  - Installed Piper in the Python virtual environment
  - Downloaded and tested a Piper voice
  - Measured Piper synthesis
  - Implemented STT-to-Piper echo
  - Installed and tested Ollama
  - Downloaded Qwen2.5 1.5B
  - Measured Qwen generation performance
  - Compared Vulkan with CPU AVX2
  - Selected CPU AVX2 based on measured results
  - Connected Whisper, Qwen and Piper
  - Added automatic Qwen warm-up
  - Implemented continuous Qwen interaction
  - Implemented spoken exit commands
  - Created and tested a Gemini API key
  - Tested Gemini independently
  - Connected Whisper, Gemini and Piper
  - Implemented continuous Gemini interaction
  - Measured Gemini response latency
  - Compared Qwen and Gemini
  - Documented STT errors and current-information limitations

  ## 23. Current status

  ReSpeaker capture                         ✅
  Voice/end-of-speech detection             ✅
  Live Faster Whisper transcription         ✅
  Whisper latency measurement               ✅
  Piper speech synthesis                    ✅
  Speaker playback                          ✅
  Offline Qwen pipeline                     ✅
  Online Gemini pipeline                    ✅
  Qwen CPU/Vulkan comparison                ✅
  Automatic Qwen warm-up                    ✅
  Continuous listen-and-answer loop         ✅
  Spoken conversation exit                  ✅
  Qwen/Gemini latency comparison            ✅ preliminary
  Conversation memory                       ❌
  STT uncertainty confirmation              ❌
  Gemini Search grounding                   ❌
  Camera/eye-contact activation             ⏳ waiting for repository
  Social Marketplace runtime integration    ⏳ parked
  Unitree integration                       ⏳ much later

  ## 24. Demonstration commands

  ### Offline Qwen demonstration

  source ~/thesis-social-robot/.venv-stt/bin/activate
  python ~/thesis-social-robot/speech_qwen_respeaker.py

  ### Online Gemini demonstration

  source ~/thesis-social-robot/.venv-stt/bin/activate

  read -s -p "Gemini API key: " GEMINI_API_KEY
  echo
  export GEMINI_API_KEY

  python ~/thesis-social-robot/speech_gemini_respeaker.py

  Do not place the Gemini API key inside the Python source file or meeting notes.

  ## 25. Main results

  ### Offline Qwen

  Average warm Qwen response: 0.56 seconds
  Average speech end → speaker start: 3.38 seconds
  Cold startup: approximately 16–26 seconds

  ### Online Gemini

  Average Gemini response: 0.69 seconds
  Average speech end → speaker start: 3.82 seconds
  No local LLM warm-up

  ### Main technical finding

  Both continuous pipelines work successfully.

  Qwen provides offline operation and slightly faster warm responses on this laptop. Gemini
  avoids local model loading and handled some incomplete inputs better, but it requires
  internet and did not automatically provide current information.

  The current latency values are preliminary baselines because the test questions and
  response lengths were not identical.

  ## 26. Questions for the next professor meeting

  1. Can I receive Atul’s camera/eye-contact repository?
  2. Is approximately 3–4 seconds of response-start latency acceptable?
  3. Should the next optimisation focus on:
      - speech endpoint detection,
      - STT accuracy,
      - Piper synthesis,
      - or response streaming?

  4. Should uncertain transcriptions cause the robot to request repetition?
  5. Is conversation memory required for the next milestone?
  6. Should Gemini Search grounding be considered for current-information questions?
  7. Which language-model mode should be preferred:
      - offline Qwen,
      - online Gemini,
      - or automatic fallback between both?

  8. Should multimodal activation be integrated before further speech-pipeline development?

