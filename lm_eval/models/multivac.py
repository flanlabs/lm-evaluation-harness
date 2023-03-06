import torch
from lm_eval.base import BaseLM
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from composer.core.precision import get_precision_context, Precision
from multivac.llm2.src.model_registry import COMPOSER_MODEL_REGISTRY
from multivac.llm2.src.mosaic_gpt import MosaicGPT, ComposerMosaicGPT
from multivac.src.tok.cl100k import Cl100kTokenizer
from typing import Optional
from lm_eval import utils
import transformers
import inspect
from omegaconf import OmegaConf as om


def build_composer_model(cfg):
    try:
        return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)
    except:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def load_state_dict_with_low_memory(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    # free up memory by placing the model in the `meta` device
    model.to(torch.device("meta"))
    keys_to_submodule = get_keys_to_submodule(model)
    for key, submodule in keys_to_submodule.items():
        # get the valye from the state_dict
        val = state_dict[key]
        # we need to substitute the parameter inside submodule,
        # remember key is composed of <name>.<subname>.<subsubname>
        # the actual submodule's parameter is stored inside the
        # last subname. If key is `in_proj.weight`, the correct field if `weight`
        param_name = key.split('.')[-1]
        param_dtype = getattr(submodule, param_name).dtype
        val = val.to(param_dtype)
        # create a new parameter
        new_val = torch.nn.Parameter(val, requires_grad=False)
        setattr(submodule, param_name, new_val)


def get_keys_to_submodule(model: nn.Module) -> dict[str, nn.Module]:
    keys_to_submodule = {}
    # iterate all submodules
    for submodule_name, submodule in model.named_modules():
        # iterate all paramters in each submobule
        for param_name, param in submodule.named_parameters():
            # param_name is organized as <name>.<subname>.<subsubname> ...
            # the more we go deep in the model, the less "subname"s we have
            splitted_param_name = param_name.split('.')
            # if we have only one subname, then it means that we reach a "leaf" submodule,
            # we cannot go inside it anymore. This is the actual parameter
            is_leaf_param = len(splitted_param_name) == 1
            if is_leaf_param:
                # we recreate the correct key
                key = f"{submodule_name}.{param_name}"
                # we associate this key with this submodule
                keys_to_submodule[key] = submodule

    return keys_to_submodule


class ComposerLLM(BaseLM):
    def __init__(
            self,
            ckpt_path: str,
            cfg_path: str,
            device: str = "cuda",
            # Can be any tokenizer whose forward method returns a dict w/ keys ['input_ids', 'attention_mask']
            batch_size: int = 2,
            precision: Optional[str] = None,
    ):
        super().__init__()

        assert isinstance(ckpt_path, str)
        assert isinstance(cfg_path, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        with open(cfg_path) as f:
            cfg = om.load(f)
        cfg.model.init_device = "meta"
        self.model = build_composer_model(cfg.model)
        tokenizer = Cl100kTokenizer()

        self.precision = precision

        # Load checkpoints
        state_dict = torch.load(ckpt_path, map_location=f'cpu')
        state_dict["state"].pop("optimizers")
        load_state_dict_with_low_memory(self.model, state_dict['state']['model'])
        self.model.to(torch.bfloat16).to(self._device)

        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size

    @property
    def eot_token_id(self):
        res = self.tokenizer.pad_token_id
        if res is None:
            res = self.tokenizer.eos_token_id
        if res is None:
            return self.tokenizer.vocab_size - 1
        return res

    @property
    def max_length(self):
        # ComposerMosaicGPT wraps around MosaicGPT as `.model` method. Thus the
        # `self.model.model`.
        if isinstance(self.model, ComposerMosaicGPT):
            return self.model.model.cfg.max_seq_len
        elif isinstance(self.model, MosaicGPT):
            return self.model.cfg.max_seq_len
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return [
            x for x in self.tokenizer(string)['input_ids'][:self.max_length]
            if x != self.tokenizer.bos_token_id
        ]

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        forward_argspec = inspect.getfullargspec(self.model.forward).args
        args = {"input_ids": inps}
        if 'key_padding_mask' in forward_argspec:
            # composer gpt uses key padding mask
            args['key_padding_mask'] = ~(inps == self.eot_token_id)
        elif 'attention_mask' in forward_argspec:
            # huggingface transformer uses attention_mask
            args['attention_mask'] = ~(inps == self.eot_token_id)

        with torch.no_grad():
            if self.precision is not None:
                with get_precision_context(self.precision):
                    res = self.model(**args)
            else:
                res = self.model(**args)

            if isinstance(res, transformers.modeling_outputs.CausalLMOutputWithPast):
                res = res.logits
            return res[:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        return super()._loglikelihood_tokens(requests, padding_length=self.max_length, padding_token=self.eot_token_id)