import sys
import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE, img_root_map
from ..smp.file import LMUDataRoot
import re
VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
DEFAULT_REGION_FEA_TOKEN = '<region_fea>'


class Ferret(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path='Ferret', root=None, vicuna_path=None, img_size=336, **kwargs):
        sys.path.append(root)
        from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from ferret.model.builder import load_pretrained_model
        from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from ferret.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from ferret.conversation import conv_templates, SeparatorStyle

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, vicuna_path, model_name)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        kwargs_default = dict(
            max_new_tokens=512, do_sample=True, num_beams=1, use_cache=True, temperature=0.001)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

        self.conv = conv_templates['ferret_v1']
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.img_size = img_size

        self.stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2

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

        # 'What is the location of <obj> in the image?'

        if task == 'Text Localization Text2Bbox':
            answer = json.loads(line['answer'])
            text = answer[0]['answer']
            cur_prompt = f'"What is the location of \"{text}\" in the image?"'
        elif task == 'Text Localization Bbox2Text':
            answer = json.loads(line['answer'])
            bbox = answer[0]['bbox']
            cur_prompt = f'What text is in area {bbox}?'
        elif task == 'Text Recognition':
            cur_prompt = 'What is text in the image and where is the text?'
        elif task == 'Key Information Extraction':
            try:
                category = re.search(r'"[^"]+"', question).group()
                cur_prompt = f'Where is the value of {category} and what is the value?'
            except:
                cur_prompt = question
        else:
            cur_prompt = f'{question} What is the answer text and where is the answer text?'

        print(f'prompt: {cur_prompt}')
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=cur_prompt))

        return msgs

    def generate_inner(self, message, dataset=None):
        conv = self.conv.copy()

        num_images = len([x for x in message if x['type'] == 'image'])
        assert num_images == 1

        qs, image = self.message_to_promptimg(message, dataset=dataset)
        try:
            with open(image, 'rb') as f:
                image = Image.open(io.BytesIO(f.read())).convert('RGB')
        except:
            print('file {} does not exist in pcache'.format(image))

        if self.model.config.mm_use_im_start_end:
            qs = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = self.DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_resize=True, do_center_crop=False, size=[self.img_size, self.img_size]
        )['pixel_values'][0]
        stopping_criteria = self.KeywordsStoppingCriteria([self.stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                stopping_criteria=[stopping_criteria],
                **self.kwargs
            )
        input_token_len = input_ids.shape[1]
        output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(self.stop_str):
            output = output[:-len(self.stop_str)]
        output = output.strip()

        return output
