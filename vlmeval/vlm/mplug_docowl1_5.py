import sys
import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class mPLUG_DocOwl1_5(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(
        self,
        model_path='MAGAer13/mplug-owl2-llama2-7b',
        anchors='grid_9',
        add_global_img=True,
        add_textual_crop_indicator=True,
        root=None,
        **kwargs
    ):
        sys.path.append(root)
        from mplug_docowl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_docowl.conversation import conv_templates, SeparatorStyle
        from mplug_docowl.model.builder import load_pretrained_model
        from mplug_docowl.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from mplug_docowl.processor import DocProcessor

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, _, _ = load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False, device='cuda'
        )
        self.doc_image_processor = DocProcessor(
            image_size=448,
            anchors=anchors,
            add_global_img=add_global_img,
            add_textual_crop_indicator=add_textual_crop_indicator
        )

        kwargs_default = dict(
            max_new_tokens=512, do_sample=False, num_beams=1,
            min_new_tokens=1, length_penalty=1, num_return_sequences=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        kwargs = cp.deepcopy(self.kwargs)
        if dataset in ['MMVet', 'LLaVABench']:
            kwargs['length_penalty'] = 0
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            kwargs['length_penalty'] = 0
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            kwargs['max_new_tokens'] = 10
        num_images = len([x for x in message if x['type'] == 'image'])
        assert num_images >= 0
        prompt_full = ''
        images = []
        if num_images == 1:
            prompt, image = self.message_to_promptimg(message, dataset=dataset)
            prompt_full += f'<|image|>{prompt}'
            images.append(image)
        else:
            for msg in message:
                if msg['type'] == 'image':
                    images.append(msg['value'])
                    prompt_full += '<|image|>'
                elif msg['type'] == 'text':
                    prompt_full += msg['value']

        image_tensor, patch_positions, text = self.doc_image_processor(images=images, query=prompt_full)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        patch_positions = patch_positions.to(self.model.device)

        conv = self.conv_templates['mplug_owl2'].copy()
        # roles = conv.roles # ('USER', 'ASSISTANT')

        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                output_hidden_states=True,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **kwargs)
        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return answer.split('</s>')[0]
