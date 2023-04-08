import whisper
import gradio as gr

def whisperTranscribe(audio):

    # Simple code

    #model = whisper.load_model("base")
    #result = model.transcribe(audio)
    #return result["text"]
    
    # Access lower-level model

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    # output the recognized text
    return result.text
  
audio_input = gr.Audio(source="microphone", type="filepath")
interface = gr.Interface(
    fn=whisperTranscribe, inputs=audio_input, outputs="text"
)
interface.launch(debug="True")
