from transformers import pipeline


transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


audio_path = "/workspaces/LLM_Transformer/src/audio/Motivation_audio.mp3"


result = transcriber(audio_path)


print(result["text"])



