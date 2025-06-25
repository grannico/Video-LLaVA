import argparse
import os

import torch

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from videollava.constants import DEFAULT_VIDEO_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from transformers import TextStreamer


def main(args):
    disable_torch_init()

    # Caricamento modello
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
        cache_dir=args.cache_dir
    )
    image_processor, video_processor = processor['image'], processor['video']

    # Imposta modalità conversazione
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode and conv_mode != args.conv_mode:
        print(f"[WARNING] inferred conv_mode={conv_mode}, but --conv-mode={args.conv_mode}. Using {args.conv_mode}")
    else:
        args.conv_mode = conv_mode

    roles = ('user', 'assistant') if "mpt" in model_name.lower() else conv_templates[args.conv_mode].roles

    # Domande automatiche da eseguire su ogni clip
    questions = [
        "Cosa succede in questa clip?",
        "Quante persone sono presenti e cosa stanno facendo?",
        "Ci sono oggetti rilevanti o azioni importanti?",
    ]

    # Per ogni clip
    for clip_path in args.file:
        ext = os.path.splitext(clip_path)[-1].lower()

        # Preprocessing
        if ext in image_ext:
            tensor = image_processor.preprocess(clip_path, return_tensors='pt')['pixel_values'][0]
        elif ext in video_ext:
            tensor = video_processor(clip_path, return_tensors='pt')['pixel_values'][0]
        else:
            print(f"[ERROR] Estensione non supportata per {clip_path}")
            continue

        tensor = tensor.to(model.device, dtype=torch.float16)

        if ext in image_ext:
            special_token = [DEFAULT_IMAGE_TOKEN]
        else:
            special_token = [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames

        print(f"\n--- Analisi automatica di {clip_path} ---")

        conv = conv_templates[args.conv_mode].copy()

        is_first = True
        for question in questions:
            inp = question
            print(f"{roles[0]}: {inp}")

            if is_first:
                if getattr(model.config, "mm_use_im_start_end", False):
                    prefix = ''.join([DEFAULT_IM_START_TOKEN + t + DEFAULT_IM_END_TOKEN for t in special_token])
                else:
                    prefix = ''.join(special_token)
                inp = prefix + '\n' + inp
                is_first = False

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[tensor],
                    do_sample=bool(args.temperature > 0),
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            print(f"{roles[1]}: {outputs}")

            if args.debug:
                print(f"\n[DEBUG] prompt:\n{prompt}\noutput:\n{outputs}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--file", nargs='+', type=str, help="Lista di file video/immagine da analizzare")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Cartella contenente clip da analizzare in ordine alfabetico")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.input_dir:
        all_files = sorted(os.listdir(args.input_dir))
        args.file = [os.path.join(args.input_dir, f)
                     for f in all_files
                     if os.path.splitext(f)[1].lower() in video_ext + image_ext]
    elif not args.file:
        parser.error("Devi specificare --file o --input-dir")

    main(args)
