# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "Spanish-F5 (Genérico)"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://jpgallegoar/F5-Spanish/model_1200000.safetensors",
    "hf://jpgallegoar/F5-Spanish/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

SPAIN_MODEL_NAME = "Spanish-F5 (España)"
SPAIN_MODEL_CFG = [
  "hf://jpgallegoar2/test/model_esp.safetensors",
  "hf://jpgallegoar/F5-Spanish/vocab.txt",
  json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

EUS_MODEL_NAME = "EUSVoice (Euskera)"
EUS_MODEL_CFG = [
  "hf://jpgallegoar2/test/model_eus_v1.safetensors",
  "hf://jpgallegoar/F5-Spanish/vocab.txt",
  json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# load models

vocoder = load_vocoder()


def load_f5tts(ckpt_path=str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
custom_ema_model, pre_custom_path = None, ""
SPAIN_ema_model, pre_spain_tts_path = None, ""
EUS_ema_model, pre_eus_tts_path = None, ""

chat_model_state = None
chat_tokenizer_state = None

#SPAIN_ema_model, pre_new_tts_path = None, ""

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    threshold,
    batch_size,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
    cfg_strength=2.0
):
    if not ref_audio_orig:
        gr.Warning("Debe subir un audio de referencia.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Debe escribir un texto a generar.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == SPAIN_MODEL_NAME:
        global SPAIN_ema_model, pre_spain_tts_path
        if pre_spain_tts_path != SPAIN_MODEL_CFG[0]:
          show_info("Cargando modelo...")
          SPAIN_ema_model = load_custom(SPAIN_MODEL_CFG[0], vocab_path=SPAIN_MODEL_CFG[1], model_cfg=json.loads(SPAIN_MODEL_CFG[2]))
          pre_spain_tts_path = SPAIN_MODEL_CFG[0]
        ema_model = SPAIN_ema_model
    elif model == EUS_MODEL_NAME:
        global EUS_ema_model, pre_eus_tts_path
        if pre_eus_tts_path != EUS_MODEL_CFG[0]:
          show_info("Cargando modelo...")
          EUS_ema_model = load_custom(EUS_MODEL_CFG[0], vocab_path=EUS_MODEL_CFG[1], model_cfg=json.loads(EUS_MODEL_CFG[2]))
          pre_eus_tts_path = EUS_MODEL_CFG[0]
        ema_model = EUS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Cargando modelo...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model
    

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
        threshold=threshold,
        batch_size=batch_size,
        cfg_strength=cfg_strength
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


with gr.Blocks() as app_credits:
    gr.Markdown("""
# Créditos

* [mrfakename](https://github.com/fakerybakery) por el [demo online original](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) por la generación inicial de fragmentos y exploración de la aplicación de podcast
* [jpgallegoar](https://github.com/jpgallegoar) por la generación de múltiples tipos de habla, chat de voz y el finetuning en español
""")
    

with gr.Blocks() as app_tts:
    gr.Markdown("# TTS por Fragmentos")
    ref_audio_input = gr.Audio(label="Audio de referencia", type="filepath")
    gen_text_input = gr.Textbox(label="Texto a generar", lines=10)
    generate_btn = gr.Button("Sintetizar", variant="primary")
    with gr.Accordion("Configuraciones Avanzadas", open=False):
        ref_text_input = gr.Textbox(
            label="Texto de Referencia",
            info="Deja en blanco para transcribir automáticamente el audio de referencia. Si ingresas texto, sobrescribirá la transcripción automática.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Eliminar Silencios",
            info="El modelo tiende a producir silencios, especialmente en audios más largos. Podemos eliminar manualmente los silencios si es necesario. Ten en cuenta que esta es una característica experimental y puede producir resultados extraños. Esto también aumentará el tiempo de generación.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Velocidad",
            minimum=0.3,
            maximum=2.0,
            value=1,
            step=0.01,
            info="Ajusta la velocidad del audio.",
        )
        cfg_strength_slider = gr.Slider(
            label="Intensidad de CFG",
            minimum=1.0,
            maximum=5.0,
            value=2.0,
            step=0.1,
            info="Controla la fuerza del Classifier-Free Guidance. Valores más altos producen audio más fiel al texto pero pueden reducir la naturalidad.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Duración del Cross-Fade (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Establece la duración del cross-fade entre clips de audio.",
        )
        steps = gr.Slider(
            label="Pasos de Inferencia",
            minimum=16,
            maximum=64,
            value=32,
            step=1,
            info="Aumentando los pasos de inferencia, se aumenta la calidad del audio de salida, pero también el tiempo que se tarda en generar el audio.",
        )
        threshold = gr.Slider(
            label="Umbral del Corrector",
            minimum=0,
            maximum=1,
            value=0.94,
            step=0.01,
            info="El umbral del corrector define la similitud que debe tener el audio de salida con el texto a generar, regenerando el fragmento de audio hasta 3 veces para maximizar este valor (mientras sea menor que el umbral).",
        )
        batch_size = gr.Slider(
            label="Duración de Cada Fragmento",
            minimum=5,
            maximum=15,
            value=15,
            step=0.5,
            info="Este valor define la longitud máxima de cada fragmento de audio sintetizado. Estos fragmentos se unen con un cross-fade para conseguir el audio final. Valor experimental.",
        )

    audio_output = gr.Audio(label="Audio Sintetizado")
    spectrogram_output = gr.Image(label="Espectrograma")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
        threshold,
        batch_size,
        cfg_strength_slider
    ):
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            threshold,
            batch_size,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
            cfg_strength=cfg_strength_slider
        )
        return audio_out, spectrogram_path, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            steps,
            speed_slider,
            threshold,
            batch_size,
            cfg_strength_slider
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input],
    )

    def clear_ref_text(audio):
        if audio is None:
            return gr.update(value="")
        return gr.update(value="")

    ref_audio_input.change(
        fn=clear_ref_text,
        inputs=[ref_audio_input],
        outputs=[ref_text_input]
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
    # Generación de Múltiples Tipos de Habla

    Esta sección te permite generar múltiples tipos de habla o las voces de múltiples personas. Ingresa tu texto en el formato mostrado a continuación, y el sistema generará el habla utilizando el tipo apropiado. Si no se especifica, el modelo utilizará el tipo de habla regular. El tipo de habla actual se usará hasta que se especifique el siguiente tipo de habla.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Entrada de Ejemplo:**                                                                      
            {Regular} Hola, me gustaría pedir un sándwich, por favor.                                                         
            {Sorprendido} ¿Qué quieres decir con que no tienen pan?                                                                      
            {Triste} Realmente quería un sándwich...                                                              
            {Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!                                                                       
            {Susurro} Solo volveré a casa y lloraré ahora.                                                                           
            {Gritando} ¿Por qué yo?!                                                                         
            """
        )

        gr.Markdown(
            """
            **Entrada de Ejemplo 2:**                                                                                
            {Speaker1_Feliz} Hola, me gustaría pedir un sándwich, por favor.                                                            
            {Speaker2_Regular} Lo siento, nos hemos quedado sin pan.                                                                                
            {Speaker1_Triste} Realmente quería un sándwich...                                                                             
            {Speaker2_Susurro} Te daré el último que estaba escondiendo.                                                                     
            """
        )

    gr.Markdown(
        "Sube diferentes clips de audio para cada tipo de habla. El primer tipo de habla es obligatorio. Puedes agregar tipos de habla adicionales haciendo clic en el botón 'Agregar Tipo de Habla'."
    )

    # Regular speech type (mandatory)
    with gr.Row() as regular_row:
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla")
            regular_insert = gr.Button("Insertar", variant="secondary")
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="Nombre del Tipo de Habla")
                delete_btn = gr.Button("Eliminar", variant="secondary")
                insert_btn = gr.Button("Insertar", variant="secondary")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Se le han agotado los tipos de habla. Considere reiniciar la aplicación.")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None

    # Update delete button clicks
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[speech_type_rows[i], speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
        )

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Textp a Generar",
        lines=10,
        placeholder="Ingresa el guión con los nombres de los hablantes (o tipos de emociones) al inicio de cada bloque, por ejemplo:\n\n{Regular} Hola, me gustaría pedir un sándwich, por favor.\n{Sorprendido} ¿Qué quieres decir con que no tienen pan?\n{Triste} Realmente quería un sándwich...\n{Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!\n{Susurro} Solo volveré a casa y lloraré ahora.\n{Gritando} ¿Por qué yo?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "None"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Configuraciones Avanzadas", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Eliminar Silencios",
            value=True,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Audio Sintetizado")

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        
        # Force using Spanish model for multi-voice
        global SPAIN_ema_model, pre_spain_tts_path
        if pre_spain_tts_path != SPAIN_MODEL_CFG[0]:
            print("Cargando modelo español...")
            SPAIN_ema_model = load_custom(SPAIN_MODEL_CFG[0], vocab_path=SPAIN_MODEL_CFG[1], model_cfg=json.loads(SPAIN_MODEL_CFG[2]))
            pre_spain_tts_path = SPAIN_MODEL_CFG[0]
        
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                gr.Warning(f"Tipo {style} no disponible, se utilizará el Regular por defecto.")
                current_style = "Regular"

            try:
                ref_audio = speech_types[current_style]["audio"]
            except KeyError:
                gr.Warning(f"Proporcione un audio de referencia para el estilo: {current_style}.")
                return [None] + [speech_types[style]["ref_text"] for style in speech_types]
            ref_text = speech_types[current_style].get("ref_text", "")

            # Generate speech for this segment using Spanish model
            audio_out, _, ref_text_out = infer(
                ref_audio, ref_text, text, SPAIN_MODEL_NAME, remove_silence, 0, show_info=print, batch_size=15
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text_out

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return [(sr, final_audio_data)] + [speech_types[style]["ref_text"] for style in speech_types]
        else:
            gr.Warning("No se ha generado ningún audio.")
            return [None] + [speech_types[style]["ref_text"] for style in speech_types]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Chat de Voz
¡Mantén una conversación con una IA usando tu voz de referencia! 
1. Sube un clip de audio de referencia y opcionalmente su transcripción.
2. Carga el modelo de chat.
3. Graba tu mensaje a través de tu micrófono.
4. La IA responderá usando la voz de referencia.
"""
    )

    if not USING_SPACES:
        load_chat_model_btn = gr.Button("Load Chat Model", variant="primary")

        chat_interface_container = gr.Column(visible=False)

        @gpu_decorator
        def load_chat_model():
            global chat_model_state, chat_tokenizer_state
            if chat_model_state is None:
                show_info = gr.Info
                show_info("Cargando modelo de chat...")
                model_name = "microsoft/phi-4"
                chat_model_state = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map="auto"
                )
                chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
                show_info("Modelo de chat cargado.")

            return gr.update(visible=False), gr.update(visible=True)

        load_chat_model_btn.click(load_chat_model, outputs=[load_chat_model_btn, chat_interface_container])

    else:
        chat_interface_container = gr.Column()

        if chat_model_state is None:
            model_name = "microsoft/phi-4"
            chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Audio de Referencia", type="filepath")
            with gr.Column():
                with gr.Accordion("Configuraciones Avanzadas", open=False):
                    remove_silence_chat = gr.Checkbox(
                        label="Eliminar Silencios",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="Texto de Referencia",
                        info="Opcional: Deja en blanco para transcribir automáticamente",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="Prompt del Sistema",
                        value="No eres un asistente de IA, eres quien el usuario diga que eres. Debes mantenerte en personaje. Mantén tus respuestas concisas ya que serán habladas en voz alta.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="Conversation")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Habla tu mensaje",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Escribe tu mensaje",
                    lines=1,
                )
                send_btn_chat = gr.Button("Enviar")
                clear_btn_chat = gr.Button("Limpiar Conversación")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "No eres un asistente de IA, eres quien el usuario diga que eres. Debes mantenerte en personaje. Mantén tus respuestas concisas ya que serán habladas en voz alta.",
                }
            ]
        )

        # Modify process_audio_input to use model and tokenizer from state
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state, ""

            conv_state.append({"role": "user", "content": text})
            history.append((text, None))

            response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)

            return history, conv_state, ""

        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, remove_silence):
            """Generate TTS audio for AI response"""
            if not history or not ref_audio:
                return None

            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None

            audio_result, _, ref_text_out = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result, ref_text_out

        def clear_conversation():
            """Reset the conversation"""
            return [], [
                {
                    "role": "system",
                    "content": "No eres un asistente de IA, eres quien el usuario diga que eres. Debes mantenerte en personaje. Mantén tus respuestas concisas ya que serán habladas en voz alta.",
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # Handle audio input
        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        # Handle text input
        text_input_chat.submit(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle send button
        send_btn_chat.click(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle clear button
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # Handle system prompt change and reset conversation
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )


with gr.Blocks() as app:
    gr.Markdown(
        """
# Spanish-F5

Esta es una interfaz web para F5 TTS, con un finetuning para poder hablar en castellano, con acento de España, y en euskera.

Implementación original:
* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

El modelo soporta el castellano con acento genérico o con acento de España.

Para los mejores resultados, intenta convertir tu audio de referencia a WAV o MP3, asegurarte de que duren entre 11 y 14 segundos, que comiencen y acaben con entre medio segundo y un segundo de silencio, y a ser posible que acabe con el final de la frase.

**NOTA: El texto de referencia será transcrito automáticamente con Whisper si no se proporciona. Para mejores resultados, mantén tus clips de referencia cortos (<15s). Asegúrate de que el audio esté completamente subido antes de generar. Se utiliza la librería num2words para convertir los números a palabras.**
"""
    )

    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        elif new_choice == SPAIN_MODEL_NAME:
            tts_model_choice = SPAIN_MODEL_NAME
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif new_choice == EUS_MODEL_NAME: 
            tts_model_choice = EUS_MODEL_NAME
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, SPAIN_MODEL_NAME, EUS_MODEL_NAME, "Custom"], label="Seleccionar modelo de TTS", value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, SPAIN_MODEL_NAME, EUS_MODEL_NAME], label="Seleccionar modelo de TTS", value=DEFAULT_TTS_MODEL
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)),
            ],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_credits],
        ["TTS Básico", "Multi-Voz", "Chat de Voz", "Créditos"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
def main(port, host, share, api, root_path):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api, root_path=root_path)


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()