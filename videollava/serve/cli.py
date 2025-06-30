import argparse
import os
import json

import torch
from transformers import TextStreamer

from videollava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import (
    tokenizer_image_token, get_model_name_from_path,
    KeywordsStoppingCriteria
)


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

    # Domande
    questions = [
        "Cosa succede in questa clip?",
        "Che modello di fucile viene mostrato o recensito?",
        "Viene descritto qualche dettaglio tecnico del fucile? Se sì, quali?",
        "Viene mostrato il funzionamento pratico del fucile? Ad esempio, il caricamento, lo sparo o il meccanismo interno?",
        "Ci sono commenti o opinioni sulle prestazioni del fucile? Se sì, quali?",
        "Sono menzionati o mostrati accessori, modifiche o personalizzazioni del fucile?",
        "Vengono mostrati test di tiro o prove pratiche? Se sì, su quali bersagli e con quali risultati?",
        "Viene fatto un confronto con altri modelli di fucile o armi simili?",
        "Ci sono indicazioni sull’utilizzo previsto del fucile? (Es. caccia, tiro sportivo, softair, difesa, collezionismo)",
        "L’utente parla di pregi e difetti? Se sì, quali vengono evidenziati?"
    ]

    results = {}

    for clip_path in args.file:
        ext = os.path.splitext(clip_path)[-1].lower()

        if ext in image_ext:
            tensor = image_processor.preprocess(clip_path, return_tensors='pt')['pixel_values'][0]
        elif ext in video_ext:
            tensor = video_processor(clip_path, return_tensors='pt')['pixel_values'][0]
        else:
            print(f"[ERROR] Estensione non supportata per {clip_path}")
            continue

        tensor = tensor.to(model.device, dtype=torch.float16)

        special_token = ([DEFAULT_IMAGE_TOKEN] if ext in image_ext
                          else [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames)

        print(f"\n--- Analisi automatica di {clip_path} ---")

        conv = conv_templates[args.conv_mode].copy()
        is_first = True

        clip_name = os.path.basename(clip_path)
        results[clip_name] = {}

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

            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if args.live else None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[tensor],
                    do_sample=bool(args.temperature > 0),
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    streamer=streamer
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

            conv.messages[-1][-1] = outputs

            if not args.live:
                print(f"{roles[1]}: {outputs}")

            if args.debug:
                print(f"\n[DEBUG] prompt:\n{prompt}\noutput:\n{outputs}\n")

            results[clip_name][question] = outputs

    # Salva il JSON
    output_json = args.output_json if args.output_json else "results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nRisultati salvati su {output_json}")

    # 🔥 Genera il prompt TXT per la LLM esterna
    output_txt = args.output_txt if args.output_txt else "summary_prompt.txt"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Ti fornirò una serie di domande seguite dalle risposte ottenute dall'analisi automatica di alcune clip video.\n")
        f.write("Il tuo compito è leggere tutte le risposte fornite e rispondere nuovamente a queste stesse domande, sintetizzando e combinando le informazioni provenienti da tutte le clip, in modo da fornire delle risposte più complete e accurate.\n")
        f.write("\n---\n\n")

        for clip, qa in results.items():
            f.write(f"🔹 Clip: {clip}\n")
            for question in questions:
                answer = qa.get(question, "Nessuna risposta")
                f.write(f"- Domanda: {question}\n")
                f.write(f"  Risposta: {answer}\n\n")
            f.write("\n---\n\n")

        f.write("🔥 Ora rispondi nuovamente alle seguenti domande tenendo conto di tutte le informazioni fornite sopra:\n\n")
        for question in questions:
            f.write(f"- {question}\n")

    print(f"Prompt per la LLM esterna salvato su {output_txt}")


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
    parser.add_argument("--live", action="store_true",
                        help="Mostra l'output in tempo reale durante la generazione (TextStreamer)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="File JSON dove salvare i risultati (default: results.json)")
    parser.add_argument("--output-txt", type=str, default=None,
                        help="File TXT dove salvare il prompt per la LLM esterna (default: summary_prompt.txt)")

    args = parser.parse_args()

    if args.input_dir:
        all_files = sorted(os.listdir(args.input_dir))
        args.file = [os.path.join(args.input_dir, f)
                     for f in all_files
                     if os.path.splitext(f)[1].lower() in video_ext + image_ext]
    elif not args.file:
        parser.error("Devi specificare --file o --input-dir")

    main(args)
