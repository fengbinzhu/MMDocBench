import json
import os
import re

import numpy as np
import regex

from vlmeval.dataset.utils.mmdocbench.parser import is_reasoning_task
from vlmeval.smp.log import get_logger

logger = get_logger('MMDocBench Evaluation')


class PredictionExtractor:
    @staticmethod
    def string_to_bbox(input_text):
        try:
            return list(eval(input_text))
        except:
            co_list = re.findall(r'[\d.]+', input_text)
            if len(co_list) == 4:
                return [float(co) for co in co_list]
            logger.debug(f'Fail to extract bbox: {input_text}')
            return []

    @staticmethod
    def get_default(default_type='dict'):
        if default_type == 'dict':
            default_result = {}
        elif default_type == '1d_array':
            default_result = []
        elif default_type == '2d_array':
            default_result = []
        else:
            raise ValueError(f'Unexpected default type: {default_type}')

        return default_result

    @staticmethod
    def form_outer_bbox(bbox_list):
        if len(bbox_list) == 0:
            return []
        outer_x0 = min([bbox[0] for bbox in bbox_list])
        outer_y0 = min([bbox[1] for bbox in bbox_list])
        outer_x1 = max([bbox[2] for bbox in bbox_list])
        outer_y1 = max([bbox[3] for bbox in bbox_list])
        return [outer_x0, outer_y0, outer_x1, outer_y1]

    @classmethod
    def group_data_to_lines(cls, ids, words, bboxes):
        block_matrix = []
        not_allocate_pool = [[idx, word, bbox] for idx, word, bbox in zip(ids, words, bboxes)]
        while len(not_allocate_pool) > 0:
            remain_pool = []
            # find first character # sort by y first and then x
            sorted_block_list = sorted(not_allocate_pool, key=lambda x: (x[2][1] ** 2, x[2][0] ** 2))
            first_block = sorted_block_list[0]

            first_block_bbox = first_block[2]
            line_block_list = [first_block]
            for i in range(1, len(sorted_block_list)):
                next_block = sorted_block_list[i]
                next_block_bbox = next_block[2]
                next_y_central = (next_block_bbox[1] + next_block_bbox[3]) * 0.5
                if first_block_bbox[1] <= next_y_central <= first_block_bbox[3]:
                    line_block_list.append(next_block)
                else:
                    remain_pool.append(next_block)
            line_central_y_list = []
            for one_block in line_block_list:
                one_block_bbox = one_block[2]
                line_central_y_list.append(one_block_bbox[1])
                line_central_y_list.append(one_block_bbox[3])
            line_central_y = np.mean(line_central_y_list)
            line_block_list = sorted(line_block_list, key=lambda x: (x[2][0] + x[2][2]) / 2)
            block_matrix.append((line_block_list, line_central_y))
            not_allocate_pool = remain_pool

        sorted_block_matrix = sorted(block_matrix, key=lambda x: x[1], reverse=False)

        line_texts = []
        line_bboxes = []

        for line_block_list, _ in sorted_block_matrix:
            line_text = ' '.join([w for _, w, _ in line_block_list])
            line_bbox = cls.form_outer_bbox([b for _, _, b in line_block_list])
            line_texts.append(line_text)
            line_bboxes.append(line_bbox)

        return line_texts, line_bboxes

    @classmethod
    def process_prediction_as_raw_string_for_textmonkey(cls, pred_str, task_name, sub_task_name, default='dict'):
        default_result = cls.get_default(default_type=default)

        prediction_results = None
        if 'Bbox2Text' in sub_task_name:
            return {'answer': pred_str, 'bbox': []}
        elif 'Text2Bbox' in sub_task_name:
            m = re.search(r'(\(\d+,\d+\)),(\(\d+,\d+\))', pred_str)
            if m:
                bbox = list(eval(m.group(1)) + eval(m.group(2)))
                return {'answer': '', 'bbox': bbox}
        elif default != '2d_array':
            words = re.findall(r'(.*?)(\(\d+,\d+\)),(\(\d+,\d+\))', pred_str)
            if words:
                word_texts = [w_tuple[0] for w_tuple in words]
                word_bboxes = [list(eval(w_tuple[1]) + eval(w_tuple[2])) for w_tuple in words]
                ids = list(range(len(word_texts)))
                line_texts, line_bboxes = cls.group_data_to_lines(ids, word_texts, word_bboxes)
            else:
                line_bboxes = []
                line_texts = [pred_str]
            if default == 'dict':
                return {'answer': '\n'.join(line_texts), 'bbox': cls.form_outer_bbox(line_bboxes)}
            return [{'answer': line_text, 'bbox': line_bbox} for line_text, line_bbox in zip(line_texts, line_bboxes)]
        elif default == '2d_array':
            return default_result

        return prediction_results

    @staticmethod
    def find_all_with_regex_list(text, regex_list):
        for one in regex_list:
            results = re.findall(one, text)
            if results:
                return results

    @classmethod
    def process_prediction_as_raw_string(cls, pred_str, task_name, default='dict', model_name='', sub_task_name=''):
        if model_name == 'textmonkey':
            return cls.process_prediction_as_raw_string_for_textmonkey(pred_str, task_name, sub_task_name, default)

        default_result = cls.get_default(default_type=default)

        prediction_results = None
        if 'Text2Bbox' in sub_task_name:
            m = re.search(r'answer":\s*(\[(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3}\])', pred_str)
            if m:
                prediction_results = {'answer': '', 'bbox': cls.string_to_bbox(m.group(1))}
        elif not is_reasoning_task(task_name) and default not in ['2d_array']:
            # simple or 1d array
            regex_list = [
                r'is\s*"([^"]+)"[^[]+(\[(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3}\])',
                r'is\s*"([^"]+)".+?\((\d+(?:,\s*\d+){3})\)',
                # answer must exist; bbox can be empty
                r'answer":\s*"([^"]+)",?\s*"bbox":\s*(\[(?:(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3})?\])',
            ]
            candidates = cls.find_all_with_regex_list(pred_str, regex_list)
            if candidates:
                candidates = list(dict.fromkeys(candidates))
                prediction_results = [{'answer': c[0], 'bbox': cls.string_to_bbox(c[1])} for c in candidates]
                if default == 'dict':
                    prediction_results = prediction_results[0]
            else:
                candidates = re.findall(r'answer":\s*"([^"]+)', pred_str)
                if candidates:
                    candidates = list(dict.fromkeys(candidates))
                    prediction_results = [{'answer': c, 'bbox': []} for c in candidates]
                    if default == 'dict':
                        prediction_results = prediction_results[0]
                elif default == 'dict':
                    prediction_results = {'answer': pred_str, 'bbox': []}
                else:
                    prediction_results = [{'answer': pred_str, 'bbox': []}]

        elif is_reasoning_task(task_name):
            # reasoning
            # answer must exit; bbox can be empty
            m = re.search(
                r'answer":\s*"([^"]+)",?\s*"bbox":\s*(\[(?:(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3})?\])',
                pred_str,
            )
            if m:
                final_answer = m.group(1)
                final_answer_bbox = cls.string_to_bbox(m.group(2))
                # text and bbox can be empty
                evidence_candidates = re.findall(
                    r'text":\s*"([^"]*)",?\s*"bbox":\s*(\[(?:(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3})?\])',
                    pred_str,
                )
                if evidence_candidates:
                    evidence_candidates = list(dict.fromkeys(evidence_candidates))
                    one_candidate = {
                        'answer': final_answer,
                        'bbox': final_answer_bbox,
                        'supporting_evidence': [
                            {'text': c[0], 'bbox': cls.string_to_bbox(c[1])} for c in evidence_candidates
                        ],
                    }
                    prediction_results = one_candidate
                else:
                    prediction_results = {
                        'answer': final_answer,
                        'bbox': final_answer_bbox,
                        'supporting_evidence': [{'text': pred_str[m.end():], 'bbox': []}],
                    }
            else:
                prediction_results = {'answer': pred_str, 'bbox': [], 'supporting_evidence': []}
        elif default == '2d_array':
            # answer and bbox can be empty
            row_candidates = re.finditer(
                (
                    r'\[(?:[^[]*answer":\s*"[^"]*",?\s*"bbox":\s*\['
                    r'(?:(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3})?\])+[^\[\]]*\]'
                ),
                pred_str,
            )
            tmp_results = []
            pointer = None
            for one_row_m in row_candidates:
                try:
                    tmp_results.append(json.loads(one_row_m.group()))
                except:
                    cells = re.findall(
                        (
                            r'answer":\s*"([^"]*)",?\s*"bbox":\s*(\[(?:(?:\d+|[0-1]\.\d+)'
                            r'(?:,\s*(?:\d+|[0-1]\.\d+)){3})?\])'
                        ),
                        one_row_m.group(),
                    )
                    row = [{'answer': c[0], 'bbox': cls.string_to_bbox(c[1])} for c in cells]
                    tmp_results.append(row)
                pointer = one_row_m.end()
            if pointer:
                remains = re.findall(
                    r'answer":\s*"([^"]*)",?\s*"bbox":\s*(\[(?:(?:\d+|[0-1]\.\d+)(?:,\s*(?:\d+|[0-1]\.\d+)){3})?\])',
                    pred_str[pointer:],
                )
                if remains:
                    remains = list(dict.fromkeys(remains))
                    last_row = [{'answer': c[0], 'bbox': cls.string_to_bbox(c[1])} for c in remains]
                    tmp_results.append(last_row)
                max_column_count = max([len(row) for row in tmp_results])
                prediction_results = []
                for row in tmp_results:
                    row = row + [{'answer': '', 'bbox': []}] * (max_column_count - len(row))
                    prediction_results.append(row)

        if not prediction_results:
            prediction_results = default_result

        return prediction_results

    @classmethod
    def process_prediction_as_json(cls, pred_str, task_name, default='dict', model_name='', sub_task_name=''):
        # clear noise which hinder json parsing
        pred_str = re.sub(r'"bbox":\s*\[empty\]', '"bbox": []', pred_str)
        if model_name == 'MiniCPM-Llama3-V-2_5':
            pred_str = re.sub(r'<box>', '[', pred_str)
            pred_str = re.sub(r'</box>\]?', ']', pred_str)

        # First match content enclosed in a JSON comment block
        json_comment_pattern = regex.compile(r'```json\s*({.*?})\s*```', regex.DOTALL)
        json_comment_match = json_comment_pattern.search(pred_str)
        json_string_candidates = []
        if json_comment_match:
            try:
                json_string = json_comment_match.group(1)
                json_dict = json.loads(json_string)
                json_string_candidates = [json_string]
            except json.JSONDecodeError:
                logger.debug(f'Failed to decode JSON from comment block: {json_comment_match.group(1)}')

        # If no JSON comment block is found, continue searching for JSON strings
        if not json_string_candidates:
            pattern = regex.compile(r'\{(?:(?R)|[^{}]*)++\}|\[(?:(?R)|[^[{}\]]*)++\]')

            json_string_candidates = pattern.findall(pred_str)

        default_result = cls.get_default(default_type=default)
        json_dict = default_result
        if json_string_candidates:
            for json_string in json_string_candidates:
                try:
                    tmp_dict = json.loads(json_string)
                    if default == '1d_array':
                        is_valid = isinstance(tmp_dict, list) and len(tmp_dict) > 0 and isinstance(tmp_dict[0], dict)
                    elif default == '2d_array':
                        is_valid = (
                            isinstance(tmp_dict, list)
                            and len(tmp_dict) > 0
                            and isinstance(tmp_dict[0], list)
                            and len(tmp_dict[0]) > 0
                            and isinstance(tmp_dict[0][0], dict)
                        )
                    elif default == 'dict':
                        is_valid = isinstance(tmp_dict, dict) and len(tmp_dict) > 0
                        if is_valid and is_reasoning_task(task_name):
                            is_valid = len(
                                {'answer', 'bbox', 'supporting_evidence'} - tmp_dict.keys()
                            ) == 0 and isinstance(tmp_dict['supporting_evidence'], list)
                    else:
                        raise ValueError(default)
                    if is_valid:
                        json_dict = tmp_dict
                        if 'Text2Bbox' in sub_task_name:
                            if 'first_occurrence' in json_dict:
                                json_dict['bbox'] = json_dict.pop('first_occurrence')
                                json_dict['answer'] = ''
                            elif 'answer' in json_dict and 'bbox' not in json_dict:
                                json_dict['bbox'] = json_dict.pop('answer')
                                json_dict['answer'] = ''
                        elif is_reasoning_task(task_name):
                            if 'supporting_evidence' in json_dict:
                                if isinstance(json_dict['supporting_evidence'], dict):
                                    json_dict['supporting_evidence'] = [json_dict['supporting_evidence']]
                                # remove string type
                                json_dict['supporting_evidence'] = [
                                    e for e in json_dict['supporting_evidence'] if isinstance(e, dict)
                                ]
                        break
                except:
                    pass

        return json_dict, json_string_candidates

    @classmethod
    def process_na(cls, pred_str, task_name, default='dict'):
        default_answer = cls.get_default(default_type=default)
        blacklist = [
            'not possible to answer',
            'cannot provide an answer',
            'there are no supporting evidences to provide',
            'I cannot answer the question',
            'the task cannot be completed',
        ]
        forgery_blacklist = ['would conclude that there are no forged texts', 'no forged texts', 'there are no']

        blacklist_to_use = blacklist if 'Forgery' not in task_name else forgery_blacklist
        for i in blacklist_to_use:
            if i in pred_str.lower():
                return default_answer

        return None

    @classmethod
    def process_prediction(cls, pred_str, task_name, default='dict', model_name='', raw_question='', sub_task_name=''):
        # check if no answer
        na = cls.process_na(pred_str, task_name, default=default)
        if na is not None:
            return na

        prediction_dict, json_string_candidates = cls.process_prediction_as_json(
            pred_str, task_name, default, model_name, sub_task_name
        )
        if not prediction_dict:
            prediction_dict = cls.process_prediction_as_raw_string(
                pred_str, task_name, default, model_name, sub_task_name
            )

        if not prediction_dict:
            candidate_str = '\n  +'.join([repr(c) for c in json_string_candidates])
            logger.debug(f'[{task_name}]Unable to parse as json. Candidates: {candidate_str}')
            logger.debug(f'[{task_name}]Unable to parse as raw string either: {repr(pred_str)}')

        return prediction_dict
