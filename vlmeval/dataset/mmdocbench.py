import pathlib
from vlmeval.dataset.utils.mmdocbench_eval import evaluate, save_metrics

from .image_base import ImageBaseDataset
from ..smp import *


class MMDocBench(ImageBaseDataset):
    TYPE = 'MMDoc'
    DATASET_URL = {
        'MMDocBench': 'https://opencompass.openxlab.space/utils/VLMEval/mmdocbench.tsv'
    }
    DATASET_MD5 = {'MMDocBench': None}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        metrics, details = evaluate(data, model_name=pathlib.Path(eval_file).stem.split('_MMDocBench')[0])
        save_file = eval_file.replace('.xlsx', '_evaluation.xlsx')
        df_score = save_metrics(metrics, details, data, save_file)
        return df_score

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))

        return msgs
