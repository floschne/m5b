#!/bin/bash

# This script is used to setup the experiment environment

setup_wandb() {
    if [ -z "$WANDB_PAT" ]; then
        echo "WANDB_PAT is not set. Please set the WANDB_PAT environment variable."
        exit 1
    fi
    echo "Setting up Weights & Biases"
    wandb login $WANDB_PAT
}

MAMBA_PREFIX=${MAMBA_PREFIX:-"${HOME}/miniforge3"}

print_python_env_info() {
    ENV_NAME=${1:-m5b}
    echo ""
    echo ""
    echo ""
    echo "##################### PYTHON ENV '${ENV_NAME}' INFO START #####################"
    echo "Python version: $(python -c 'import sys; print(sys.version)')"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "Flash Attention 2 Support: $(python -c 'import importlib.util; import torch; fattn="flash_attn_2_cuda";print(importlib.util.find_spec(fattn) is not None)')"
    echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
    echo "Datasets version: $(python -c 'import datasets; print(datasets.__version__)')"
    echo "Lightning version: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"
    echo "##################### PYTHON ENV '${ENV_NAME}' INFO END #####################"
    echo ""
    echo ""
    echo ""
}

install_mamba() {
    echo "Installing mamba"
    # check if mamba is already installed
    if command -v mamba &>/dev/null; then
        echo "Mamba is already installed. Skipping installation."
        return
    fi
    cd $HOME
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b
}

init_mamba() {
    # >>> conda initialize >>>
    CONDA_SETUP="${MAMBA_PREFIX}/bin/conda 'shell.bash' 'hook'"
    __conda_setup=$("$CONDA_SETUP" 2>/dev/null)
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$MAMBA_PREFIX/etc/profile.d/conda.sh" ]; then
            . "$MAMBA_PREFIX/etc/profile.d/conda.sh"
        else
            export PATH="$MAMBA_PREFIX/bin:$PATH"
        fi
    fi
    unset __conda_setup

    if [ -f "$MAMBA_PREFIX/etc/profile.d/mamba.sh" ]; then
        . "$MAMBA_PREFIX/etc/profile.d/mamba.sh"
    fi
    # <<< conda initialize <<<
}

activate_mamba_env() {
    ENV_NAME=${1:-m5b}
    mamba activate $ENV_NAME || {
        echo "Failed to activate Conda environment $ENV_NAME! Exiting."
        exit 1
    }
}

create_mamba_env() {
    ENV_NAME=${1:-m5b}
    echo "Creating up python environment with mamba"

    init_mamba

    echo "Creating python environment $ENV_NAME"

    # check if the environment already exists
    if mamba env list | grep -q $ENV_NAME; then
        echo "Environment $ENV_NAME already exists. Skipping creation."
        return
    fi

    case $ENV_NAME in
    "m5b")
        if [ ! -f $PWD/environment.yml ]; then
            echo "environment.yml file not found in the current directory. Exiting."
            exit 1
        fi

        mamba create -y -n $ENV_NAME python=3.10

        mamba env update -n $ENV_NAME -f $PWD/environment.yml

        activate_mamba_env $ENV_NAME

        pip install --no-cache-dir einops ninja
        pip install --no-cache-dir --no-deps multilingual-clip open_clip_torch
        pip install flash-attn --no-build-isolation
        ;;
    "m5b-cogvlm")
        if [ ! -f $PWD/cogvlm_environment.yml ]; then
            echo "cogvlm_environment.yml file not found in the current directory. Exiting."
            exit 1
        fi

        mamba create -y -n $ENV_NAME python=3.10

        mamba env update -n $ENV_NAME -f $PWD/cogvlm_environment.yml

        activate_mamba_env $ENV_NAME

        # see https://huggingface.co/THUDM/cogvlm-chat-hf
        pip install -U --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0 torchvision xformers
        pip install "transformers>=4.35.0" sentencepiece==0.1.99 einops==0.7.0 datasets lightning torchmetrics spacy spacy-legacy pycocoevalcap evaluate
        pip install "accelerate>=0.24.1"
        pip install sacrebleu rouge_score "spacy[ja]" "spacy[th]" "spacy[zh]" nltk absl-py bert_score

        ;;
    "m5b-llava")
        if [ ! -f $PWD/llava_environment.yml ]; then
            echo "llava_environment.yml file not found in the current directory. Exiting."
            exit 1
        fi

        mamba create -y -n $ENV_NAME python=3.10

        mamba env update -n $ENV_NAME -f $PWD/llava_environment.yml

        activate_mamba_env $ENV_NAME

        pip install --upgrade pip # enable PEP 660 support
        pip install flash-attn --no-build-isolation

        #see https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install
        git clone https://github.com/haotian-liu/LLaVA.git
        cd LLaVA
        pip install -e .
        # eval deps
        pip install lightning torchmetrics spacy spacy-legacy evaluate
        pip install pycocoevalcap datasets sacrebleu rouge_score "spacy[ja]" "spacy[th]" "spacy[zh]" nltk absl-py bert_score
        ;;
    "m5b-omnilmm")
        if [ ! -f $PWD/omnilmm_environment.yml ]; then
            echo "omnilmm_environment.yml file not found in the current directory. Exiting."
            exit 1
        fi

        mamba create -y -n $ENV_NAME python=3.10

        mamba env update -n $ENV_NAME -f $PWD/omnilmm_environment.yml

        activate_mamba_env $ENV_NAME

        #see https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install
        git clone https://github.com/OpenBMB/OmniLMM.git
        pip install -r OmniLMM/requirements.txt

        # eval deps
        pip install lightning torchmetrics spacy spacy-legacy evaluate
        pip install pycocoevalcap datasets sacrebleu rouge_score "spacy[ja]" "spacy[th]" "spacy[zh]" nltk absl-py bert_score

        pip install flash-attn --no-build-isolation

        # since OmniLMM doesn't provide a setup.py, we need to "install" it manually
        mkdir -p $PWD/omnilmm
        ln -s $(realpath ./OmniLMM/omnilmm) $PWD/omnilmm
        ln -s $(realpath ./OmniLMM/chat.py) $PWD/omnilmm_chat.py

        ;;
    "m5b-qwenvl")
        if [ ! -f $PWD/qwenvl_environment.yml ]; then
            echo "qwenvl_environment.yml file not found in the current directory. Exiting."
            exit 1
        fi

        mamba create -y -n $ENV_NAME python=3.10

        mamba env update -n $ENV_NAME -f $PWD/llava_environment.yml # llava is intentional here

        activate_mamba_env $ENV_NAME

        # eval deps
        pip install lightning torchmetrics spacy spacy-legacy evaluate
        pip install pycocoevalcap datasets sacrebleu rouge_score "spacy[ja]" "spacy[th]" "spacy[zh]" nltk absl-py bert_score

        pip install tiktoken "transformers_stream_generator==0.0.4"
        ;;
    "m5b-yivl")
        if [ ! -f $PWD/yivl_environment.yml ]; then
            echo "yivl_environment.yml file not found in the current directory. Exiting."
            exit 1
        fi

        mamba create -y -n $ENV_NAME python=3.10

        mamba env update -n $ENV_NAME -f $PWD/llava_environment.yml # llava is intentional here

        activate_mamba_env $ENV_NAME

        #see https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install
        git clone https://github.com/01-ai/Yi.git
        cd Yi
        cat >./yi-vl-local-import.patch <<EOL
diff --git a/VL/llava/mm_utils.py b/VL/llava/mm_utils.py
index 1bb61c7..af2c997 100644
--- a/VL/llava/mm_utils.py
+++ b/VL/llava/mm_utils.py
@@ -2,8 +2,8 @@ import base64
 from io import BytesIO

 import torch
-from llava.model import LlavaLlamaForCausalLM
-from llava.model.constants import IMAGE_TOKEN_INDEX
+from .model import LlavaLlamaForCausalLM
+from .model.constants import IMAGE_TOKEN_INDEX
 from PIL import Image
 from transformers import AutoTokenizer, StoppingCriteria

diff --git a/VL/llava/model/llava_arch.py b/VL/llava/model/llava_arch.py
index 8815515..f5eefc6 100644
--- a/VL/llava/model/llava_arch.py
+++ b/VL/llava/model/llava_arch.py
@@ -17,7 +17,7 @@ import os
 from abc import ABC, abstractmethod

 import torch
-from llava.model.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, key_info
+from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, key_info

 from .clip_encoder.builder import build_vision_tower
 from .multimodal_projector.builder import build_vision_projector
diff --git a/VL/llava/model/llava_llama.py b/VL/llava/model/llava_llama.py
index ebacc3a..688b9d2 100644
--- a/VL/llava/model/llava_llama.py
+++ b/VL/llava/model/llava_llama.py
@@ -140,7 +140,7 @@ class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
         past_key_values=None,
         attention_mask=None,
         inputs_embeds=None,
-        **kwargs
+        **kwargs,
     ):
         if past_key_values:
             input_ids = input_ids[:, -1:]
EOL
        git apply yi-vl-local-import.patch

        # since Yi VL doesn't provide a setup.py, we need to "install" it manually
        mkdir -p $PWD/yi
        cp -r "$(realpath ./VL)/*" $PWD/yi

        # eval deps
        pip install lightning torchmetrics spacy spacy-legacy evaluate sentencepiece accelerate
        pip install pycocoevalcap datasets sacrebleu rouge_score "spacy[ja]" "spacy[th]" "spacy[zh]" nltk absl-py bert_score

        pip install flash-attn --no-build-isolation
        echo Downloading models via git lfs... This takes a while and looks as if its unreponsive!

        cd $PWD/yi
        git lfs install
        print Yi-VL-6B
        git clone https://huggingface.co/01-ai/Yi-VL-6B
        print Yi-VL-34B
        git clone https://huggingface.co/01-ai/Yi-VL-34B
        ;;
    *)
        echo "Unknown environment name: $ENV_NAME"
        exit 1
        ;;
    esac

    print_python_env_info $ENV_NAME
}

remove_env_mamba() {
    ENV_NAME=${1:-m5b}
    echo "Removing python environment with mamba"

    init_mamba

    mamba env remove -n $ENV_NAME
}

if [ $# -eq 0 ]; then
    install_mamba
    create_mamba_env m5b
    create_mamba_env m5b-llava
    create_mamba_env m5b-yivl
    create_mamba_env m5b-cogvlm
    create_mamba_env m5b-qwenvl
    create_mamba_env m5b-omnilmm
else
    for arg in "$@"; do
        case $arg in
        "setup-wandb")
            setup_wandb
            ;;
        "install-mamba")
            install_mamba
            ;;
        "mamba-env-create")
            create_mamba_env $2
            ;;
        "mamba-env-create-force")
            remove_env_mamba $2
            create_mamba_env $2
            ;;
        "print-python-env-info")
            init_mamba
            activate_mamba_env $2
            print_python_env_info $2
            ;;
        *)
            echo "Invalid argument: $arg"
            ;;
        esac
    done
fi
