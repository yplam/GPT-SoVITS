import argparse
import os
import pandas as pd
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()


def batch_synthesize(dry_run, model_path, target_csv, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df = pd.read_excel(target_csv, engine='openpyxl')
    columns_to_extract = ['model', 'content', 'tone', 'fid']
    extracted_columns = df.iloc[:, [3, 4, 6, 7]]
    extracted_columns.columns = columns_to_extract
    grouped = extracted_columns.groupby('model')
    global_ref_dir = os.path.join(model_path, '参考音频')
    for name, group in grouped:
        model_dir = os.path.join(model_path, name)
        if not os.path.exists(model_dir):
            print(f'Model {name} not found in {model_path}')
            continue
        # find .ckpt and .pth files
        ckpt_file = None
        pth_file = None
        for file in os.listdir(model_dir):
            if file.endswith('.ckpt'):
                ckpt_file = file
            elif file.endswith('.pth'):
                pth_file = file
        if ckpt_file is None or pth_file is None:
            print(f'Model {name} does not contain both .ckpt and .pth files')
            continue
        ref_dir = os.path.join(model_dir, "参考音频")
        tones = {}
        if os.path.exists(ref_dir):
            for file in os.listdir(ref_dir):
                if file.endswith('.wav'):
                    sps = file.split('-', 1)
                    ref_tone = sps[0]
                    ref_text, _ = os.path.splitext(sps[1])
                    tones[ref_tone] = {'text': ref_text, 'file': os.path.join(ref_dir, file)}
                    # print(f'Reference audio, Tone: {ref_tone}, Text: {ref_text}')

        for file in os.listdir(global_ref_dir):
            if file.endswith('.wav'):
                sps = file.split('-', 2)
                if len(sps) < 3:
                    print(f'too short file name: {file}')
                    continue
                model_name = sps[0]
                ref_tone = sps[1]
                ref_text, _ = os.path.splitext(sps[2])
                if model_name != name:
                    continue
                tones[ref_tone] = {'text': ref_text, 'file': os.path.join(global_ref_dir, file)}

        if '说话' not in tones:
            print(f'No reference audio found in {ref_dir}')
            continue

        if not dry_run:
            print(f'Character: {name}, gpt: {ckpt_file}, sovits: {pth_file}, tones: {tones} \n')
            # Change model weights
            change_gpt_weights(gpt_path=os.path.join(model_dir, ckpt_file))
            change_sovits_weights(sovits_path=os.path.join(model_dir, pth_file))
            ref_audio_path = tones['说话']['file']
            ref_text = tones['说话']['text']

            for index, row in group.iterrows():
                try:
                    row["fid"] = int(row["fid"])
                except:
                    print(f'ID: {row["fid"]} is not an integer')
                    continue
                print(f'ID: {int(row["fid"])}, Text: {row["content"]}, Tone: {row["tone"]} \n')
                if row["tone"] in tones:
                    ref_audio_path = tones[row["tone"]]['file']
                    ref_text = tones[row["tone"]]['text']

                for i in range(1, 11):
                    output_wav_path = os.path.join(output_path, f'{row["fid"]}_{i}.wav')
                    if os.path.exists(output_wav_path):
                        print(f'Audio file {output_wav_path} already exists')
                        continue

                    # Synthesize audio
                    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path,
                                                   prompt_text=ref_text,
                                                   prompt_language=i18n('中文'),
                                                   how_to_cut=i18n('凑四句一切'),
                                                   text=row["content"],
                                                   text_language=i18n('中文'), top_k=10, top_p=1, temperature=1)

                    result_list = list(synthesis_result)

                    if result_list:
                        last_sampling_rate, last_audio_data = result_list[-1]
                        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
                        print(f"Audio saved to {output_wav_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--dry_run', required=False, default=False)
    parser.add_argument('--model_path', required=False, default="models", help="模型总目录，包含多个模型名称的子目录，每个子目录包含一个"
                                                                               "ckpt文件，一个pth文件，以及一个参考音频目录")
    parser.add_argument('--target_csv', required=True,
                        help="目标文本csv文件，C列为模型名称，F列为目标文本，H为语调，I为输出ID")
    parser.add_argument('--output_path', required=True, help="输出目录")

    args = parser.parse_args()

    batch_synthesize(args.dry_run, args.model_path, args.target_csv, args.output_path)


if __name__ == '__main__':
    main()
