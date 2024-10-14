import torch

import warnings
from .base import BaseModel
from ..dataset import DATASET_TYPE, img_root_map
from ..smp.file import LMUDataRoot
import json
import sys
import os.path as osp


class TextMonkey(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='echo840/Monkey', root=None, **kwargs):
        sys.path.append(root)
        from monkey_model.modeling_textmonkey import TextMonkeyLMHeadModel
        from monkey_model.tokenization_qwen import QWenTokenizer
        from monkey_model.configuration_monkey import MonkeyConfig

        assert model_path is not None
        self.model_path = model_path
        config = MonkeyConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = QWenTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.IMG_TOKEN_SPAN = config.visual['n_queries']
        self.tokenizer = tokenizer
        model = TextMonkeyLMHeadModel.from_pretrained(model_path,
                                                      config=config,
                                                      device_map='cpu',
                                                      trust_remote_code=True).eval()
        self.model = model.cuda()
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

        self.img_root = LMUDataRoot()

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MMDoc':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        ROOT = LMUDataRoot()
        self.img_root = osp.join(ROOT, 'images', img_root_map(dataset))

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)

        question = line['raw_question']
        task = line['task']

        if task == 'Text Recognition':
            cur_prompt = f'<img>{tgt_path[0]}</img> OCR with grounding:'
        elif task == 'Table Recognition':
            cur_prompt = f'<img>{tgt_path[0]}</img> Convert the table in this image to json format. Answer:'
        elif task == 'Text Localization Text2Bbox':
            answer = json.loads(line['answer'])
            text = answer[0]['answer']
            cur_prompt = f'<img>{tgt_path[0]}</img> <ref>{text}</ref>'
        elif task == 'Text Localization Bbox2Text':
            answer = json.loads(line['answer'])
            bbox = answer[0]['bbox']
            x1, y1, x2, y2 = bbox
            cur_prompt = f'<img>{tgt_path[0]}</img> <ref>This</ref> <box>({x1},{y1}),({x2},{y2})</box>is'
        elif task == 'Cell Localization':
            cur_prompt = (
                f'<img>{tgt_path[0]}</img> Start counting rows from header. '
                f'{question} Provide the location coordinates of the answer when answering the question: '
            )
        else:
            cur_prompt = (
                f'<img>{tgt_path[0]}</img> {question}. '
                f'Provide the location coordinates of the answer when answering the question: '
            )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=cur_prompt))

        return msgs

    def generate_vanilla(self, prompt):

        input_ids = self.tokenizer(prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        output_ids = self.model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.kwargs['max_new_tokens'],
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
        )
        response = self.tokenizer.decode(
            output_ids[0][input_ids.size(1):].cpu(),
            skip_special_tokens=True
        ).strip()
        return response

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if dataset is None:
            return self.generate_vanilla(prompt)
        assert isinstance(dataset, str)
        return self.generate_vanilla(prompt)
