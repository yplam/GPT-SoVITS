import argparse
import json
import os
from subprocess import Popen
import shutil

import yaml

from tools.my_utils import load_audio, check_for_existance, check_details

def dataset_formatting(model):
    model = model.strip()
    opt_dir = 'logs/%s' % model
    if os.path.exists(opt_dir):
        shutil.rmtree(opt_dir)
    inp_dir = 'input/%s' % model
    config = {
        "inp_text": inp_dir+"/label.list",
        "inp_wav_dir": inp_dir,
        "exp_name": model,
        "opt_dir": opt_dir,
        "bert_pretrained_dir": 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
        "is_half": 'False',
        "i_part": '0',
        "all_parts": '1',
        "_CUDA_VISIBLE_DEVICES": '0',
    }
    os.environ.update(config)
    cmd = '/usr/bin/python GPT_SoVITS/prepare_datasets/1-get-text.py'
    print(cmd)
    print(config)
    p = Popen(cmd, shell=True)
    p.wait()
    txt_path = "%s/2-name2text-0.txt" % opt_dir
    txt_new_path = "%s/2-name2text.txt" % opt_dir
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)
    os.rename(txt_path, txt_new_path)
    print("文本进程成功")

    config = {
        "inp_text": inp_dir + "/label.list",
        "inp_wav_dir": inp_dir,
        "exp_name": model,
        "opt_dir": opt_dir,
        "cnhubert_base_dir": 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
        "is_half": 'False',
        "i_part": '0',
        "all_parts": '1',
        "_CUDA_VISIBLE_DEVICES": '0',
    }
    os.environ.update(config)
    cmd = '/usr/bin/python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
    print(cmd)
    print(config)
    p = Popen(cmd, shell=True)
    p.wait()
    print("SSL提取进程结束")

    config = {
        "inp_text": inp_dir + "/label.list",
        "exp_name": model,
        "opt_dir": opt_dir,
        "pretrained_s2G": 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth',
        "s2config_path": 'GPT_SoVITS/configs/s2.json',
        "is_half": 'False',
        "i_part": '0',
        "all_parts": '1',
        "_CUDA_VISIBLE_DEVICES": '0',
    }
    os.environ.update(config)
    cmd = '/usr/bin/python GPT_SoVITS/prepare_datasets/3-get-semantic.py'
    print(cmd)
    print(config)
    p = Popen(cmd, shell=True)
    p.wait()
    opt = ["item_name\tsemantic_audio"]
    path_semantic = "%s/6-name2semantic.tsv" % opt_dir
    semantic_path = "%s/6-name2semantic-0.tsv" % opt_dir
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(semantic_path)
    with open(semantic_path, "r", encoding="utf8") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(semantic_path)
    with open(path_semantic, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")


def train_sovits(model):
    with open("GPT_SoVITS/configs/s2.json") as f:
        data = f.read()
        data = json.loads(data)

    model = model.strip()
    exp_root = 'logs'
    exp_name = model
    s2_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
    if check_for_existance([s2_dir], is_train=True):
        check_details([s2_dir], is_train=True)
    data["train"]["fp16_run"] = False
    batch_size = 4
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = 8
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth'
    data["train"]["pretrained_s2D"] = 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth'
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = 8
    data["train"]["gpu_numbers"] = "0"
    data["model"]["version"] = "v2"
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = "SoVITS_weights_v2"
    data["name"] = exp_name
    data["version"] = "v2"
    tmp_config_path = "TEMP/tmp_s2.json"
    with open(tmp_config_path, "w") as f:
        f.write(json.dumps(data))

    cmd = '/usr/bin/python GPT_SoVITS/s2_train.py --config "%s"'%tmp_config_path
    print(cmd)
    print('SoVITS训练开始')
    p = Popen(cmd, shell=True)
    p.wait()
    print('SoVITS训练结束')

def train_gpt(model):
    version = "v2"
    model = model.strip()
    exp_root = 'logs'
    exp_name = model
    with open("GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    s1_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
    if check_for_existance([s1_dir], is_train=True):
        check_details([s1_dir], is_train=True)
    data["train"]["precision"] = "32"
    data["train"]["batch_size"] = 6
    data["train"]["epochs"] = 15
    data["pretrained_s1"] = 'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
    data["train"]["save_every_n_epoch"] = 15
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = True
    data["train"]["half_weights_save_dir"] = 'GPT_weights_v2'
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
    data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
    data["output_dir"] = "%s/logs_s1" % s1_dir
    # data["version"]=version

    os.environ["_CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["hz"] = "25hz"
    tmp = "TEMP"
    tmp_config_path = "%s/tmp_s1.yaml" % tmp
    with open(tmp_config_path, "w") as f:
        f.write(yaml.dump(data, default_flow_style=False))
    # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
    cmd = '/usr/bin/python GPT_SoVITS/s1_train.py --config_file "%s" ' % tmp_config_path
    print(cmd)
    print('GPT训练开始')
    p = Popen(cmd, shell=True)
    p.wait()
    print('GPT训练结束')


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--model', required=False, default="", help="模型名称，将根据此名称拼接路径")
    args = parser.parse_args()
    dataset_formatting(args.model)
    train_sovits(args.model)
    train_gpt(args.model)


if __name__ == '__main__':
    main()