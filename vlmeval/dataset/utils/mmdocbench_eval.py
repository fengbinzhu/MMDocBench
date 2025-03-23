import itertools
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from vlmeval.dataset.utils.mmdocbench.extractor import PredictionExtractor
from vlmeval.dataset.utils.mmdocbench.parser import BboxParser, TextParser, is_reasoning_task, normalize_text
from vlmeval.smp.log import get_logger

logger = get_logger('MMDocBench Evaluation')


CAPABILITY_NAME = 'category'

paper_order = [
    ('Text Recognition', 'TextOCR'),
    ('Text Recognition', 'BookOCR'),
    ('Table Recognition', 'FinTabNet'),
    ('Table Recognition', 'PubTables-1M'),
    ('Text Localization', 'Text2Bbox'),
    ('Text Localization', 'Bbox2Text'),
    ('Cell Localization', 'FinTabNet'),
    ('Cell Localization', 'PubTables-1M'),
    ('Key Information Extraction', 'SROIE'),
    ('Key Information Extraction', 'WildReceipt'),
    ('Key Information Extraction', 'CORD'),
    ('Document Forgery Detection', 'T-SROIE'),
    ('Document Forgery Detection', 'DocTamper'),
    ('Document Question Answering', 'DocVQA'),
    ('Document Question Answering', 'WTQ'),
    ('Document Question Answering', 'TAT-DQA'),
    ('Chart Question Answering', 'ChartQA'),
    ('Chart Question Answering', 'CharXiv'),
    ('Infographic Question Answering', 'InfographicVQA'),
    ('Arithmetic Reasoning', 'DUDE'),
    ('Arithmetic Reasoning', 'WTQ'),
    ('Arithmetic Reasoning', 'TAT-DQA'),
    ('Arithmetic Reasoning', 'CharXiv'),
    ('Arithmetic Reasoning', 'InfographicVQA'),
    ('Logical Reasoning', 'DUDE'),
    ('Logical Reasoning', 'WTQ'),
    ('Logical Reasoning', 'TAT-DQA'),
    ('Logical Reasoning', 'CharXiv'),
    ('Logical Reasoning', 'InfographicVQA'),
    ('Spatial Reasoning', 'DUDE'),
    ('Spatial Reasoning', 'WTQ'),
    ('Spatial Reasoning', 'CharXiv'),
    ('Spatial Reasoning', 'InfographicVQA'),
    ('Comparison', 'DUDE'),
    ('Comparison', 'WTQ'),
    ('Comparison', 'TAT-DQA'),
    ('Comparison', 'CharXiv'),
    ('Comparison', 'InfographicVQA'),
    ('Sorting', 'DUDE'),
    ('Sorting', 'WTQ'),
    ('Sorting', 'TAT-DQA'),
    ('Sorting', 'CharXiv'),
    ('Sorting', 'InfographicVQA'),
    ('Counting', 'DUDE'),
    ('Counting', 'WTQ'),
    ('Counting', 'TAT-DQA'),
    ('Counting', 'CharXiv'),
    ('Counting', 'InfographicVQA'),
]

task_order = [
    'Text Recognition',
    'Table Recognition',
    'Text Localization',
    'Cell Localization',
    'Key Information Extraction',
    'Document Forgery Detection',
    'Document Question Answering',
    'Chart Question Answering',
    'Infographic Question Answering',
    'Arithmetic Reasoning',
    'Logical Reasoning',
    'Spatial Reasoning',
    'Comparison',
    'Sorting',
    'Counting',
]


def overlap_iou(r1, r2):
    if not r1 or not r2:
        return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(r1[0], r2[0])
    y_a = max(r1[1], r2[1])
    x_b = min(r1[2], r2[2])
    y_b = min(r1[3], r2[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
    box_b_area = (r2[2] - r2[0]) * (r2[3] - r2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / max(float(box_a_area + box_b_area - inter_area), 0.00001)

    # return the intersection over union value
    return iou


def exact_match(prediction, truth):
    return int(prediction == truth)


def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return round(2 * (prec * rec) / (prec + rec), 2)


class HungarianMatcher:
    @staticmethod
    def compute_1d_rewards(pred_results, gold_results, reward_fn):
        pre_computed_rewards = np.zeros((len(pred_results), len(gold_results)))
        for pred_idx, gold_idx in itertools.product(range(len(pred_results)), range(len(gold_results))):
            reward = reward_fn(pred_results[pred_idx], gold_results[gold_idx])
            pre_computed_rewards[(pred_idx, gold_idx)] = reward
        return pre_computed_rewards

    @staticmethod
    def align_1d(reward_lookup):
        sequence1_indices, sequence2_indices = linear_sum_assignment(reward_lookup, maximize=True)
        seq_map = {i: j for i, j in zip(sequence1_indices, sequence2_indices)}
        return seq_map


class Grits:
    @staticmethod
    def initialize_DP(sequence1_length, sequence2_length):
        """
        Helper function to initialize dynamic programming data structures.
        """
        # Initialize DP tables
        scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
        pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))

        # Initialize pointers in DP table
        for seq1_idx in range(1, sequence1_length + 1):
            pointers[seq1_idx, 0] = -1

        # Initialize pointers in DP table
        for seq2_idx in range(1, sequence2_length + 1):
            pointers[0, seq2_idx] = 1

        return scores, pointers

    @staticmethod
    def traceback(pointers):
        """
        Dynamic programming traceback to determine the aligned indices
        between the two sequences.

        Traceback convention: -1 = up, 1 = left, 0 = diag up-left
        """
        seq1_idx = pointers.shape[0] - 1
        seq2_idx = pointers.shape[1] - 1
        aligned_sequence1_indices = []
        aligned_sequence2_indices = []
        while not (seq1_idx == 0 and seq2_idx == 0):
            if pointers[seq1_idx, seq2_idx] == -1:
                seq1_idx -= 1
            elif pointers[seq1_idx, seq2_idx] == 1:
                seq2_idx -= 1
            else:
                seq1_idx -= 1
                seq2_idx -= 1
                aligned_sequence1_indices.append(seq1_idx)
                aligned_sequence2_indices.append(seq2_idx)

        aligned_sequence1_indices = aligned_sequence1_indices[::-1]
        aligned_sequence2_indices = aligned_sequence2_indices[::-1]

        return aligned_sequence1_indices, aligned_sequence2_indices

    @classmethod
    def align_1d(cls, sequence1, sequence2, reward_lookup):
        '''
        Dynamic programming alignment between two sequences,
        with memoized rewards.

        Sequences are represented as indices into the rewards lookup table.

        Traceback convention: -1 = up, 1 = left, 0 = diag up-left
        '''
        sequence1_length = len(sequence1)
        sequence2_length = len(sequence2)

        scores, pointers = cls.initialize_DP(sequence1_length, sequence2_length)

        for seq1_idx in range(1, sequence1_length + 1):
            for seq2_idx in range(1, sequence2_length + 1):
                reward = reward_lookup[sequence1[seq1_idx - 1] + sequence2[seq2_idx - 1]]
                diag_score = scores[seq1_idx - 1, seq2_idx - 1] + reward
                skip_seq2_score = scores[seq1_idx, seq2_idx - 1]
                skip_seq1_score = scores[seq1_idx - 1, seq2_idx]

                max_score = max(diag_score, skip_seq1_score, skip_seq2_score)
                scores[seq1_idx, seq2_idx] = max_score
                if diag_score == max_score:
                    pointers[seq1_idx, seq2_idx] = 0
                elif skip_seq1_score == max_score:
                    pointers[seq1_idx, seq2_idx] = -1
                else:  # skip_seq2_score == max_score
                    pointers[seq1_idx, seq2_idx] = 1

        # Traceback
        sequence1_indices, sequence2_indices = cls.traceback(pointers)

        seq_map = {i: j for i, j in zip(sequence1_indices, sequence2_indices)}

        score = scores[-1, -1]

        return seq_map, score

    @classmethod
    def align_2d_outer(cls, true_shape, pred_shape, reward_lookup):
        '''
        Dynamic programming matrix alignment posed as 2D
        sequence-of-sequences alignment:
        Align two outer sequences whose entries are also sequences,
        where the match reward between the inner sequence entries
        is their 1D sequence alignment score.

        Traceback convention: -1 = up, 1 = left, 0 = diag up-left
        '''

        scores, pointers = cls.initialize_DP(true_shape[0], pred_shape[0])

        for row_idx in range(1, true_shape[0] + 1):
            for col_idx in range(1, pred_shape[0] + 1):
                _, reward = cls.align_1d(
                    [(row_idx - 1, tcol) for tcol in range(true_shape[1])],
                    [(col_idx - 1, prow) for prow in range(pred_shape[1])],
                    reward_lookup,
                )
                diag_score = scores[row_idx - 1, col_idx - 1] + reward
                same_row_score = scores[row_idx, col_idx - 1]
                same_col_score = scores[row_idx - 1, col_idx]

                max_score = max(diag_score, same_col_score, same_row_score)
                scores[row_idx, col_idx] = max_score
                if diag_score == max_score:
                    pointers[row_idx, col_idx] = 0
                elif same_col_score == max_score:
                    pointers[row_idx, col_idx] = -1
                else:
                    pointers[row_idx, col_idx] = 1

        score = scores[-1, -1]

        aligned_true_indices, aligned_pred_indices = cls.traceback(pointers)

        return aligned_true_indices, aligned_pred_indices, score

    @staticmethod
    def compute_fscore(num_true_positives, num_true, num_positives):
        """
        Compute the f-score or f-measure for a collection of predictions.

        Conventions:
        - precision is 1 when there are no predicted instances
        - recall is 1 when there are no true instances
        - fscore is 0 when recall or precision is 0
        """
        if num_positives > 0:
            precision = num_true_positives / num_positives
        else:
            precision = 1
        if num_true > 0:
            recall = num_true_positives / num_true
        else:
            recall = 1

        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0

        return fscore, precision, recall

    @classmethod
    def find_2d_alignment(cls, true_cell_grid, pred_cell_grid, reward_function):
        """
        Factored 2D-MSS: Factored two-dimensional most-similar substructures

        This is a polynomial-time heuristic to computing the 2D-MSS of two matrices,
        which is NP hard.

        A substructure of a matrix is a subset of its rows and its columns.

        The most similar substructures of two matrices, A and B, are the substructures
        A' and B', where the sum of the similarity over all corresponding entries
        A'(i, j) and B'(i, j) is greatest.
        """
        true_row_count = len(true_cell_grid)
        pred_row_count = len(pred_cell_grid)

        true_column_count = len(true_cell_grid[0]) if true_row_count else 0
        pred_column_count = len(pred_cell_grid[0]) if pred_row_count else 0

        pre_computed_rewards = {}
        transpose_rewards = {}
        for trow, tcol, prow, pcol in itertools.product(
            range(true_row_count), range(true_column_count), range(pred_row_count), range(pred_column_count)
        ):
            reward = reward_function(
                pred_cell_grid[prow][pcol] if pcol < len(pred_cell_grid[prow]) else {}, true_cell_grid[trow][tcol]
            )

            pre_computed_rewards[(trow, tcol, prow, pcol)] = reward
            transpose_rewards[(tcol, trow, pcol, prow)] = reward

        true_row_nums, pred_row_nums, row_pos_match_score = cls.align_2d_outer(
            [true_row_count, true_column_count], [pred_row_count, pred_column_count], pre_computed_rewards
        )

        true_column_nums, pred_column_nums, col_pos_match_score = cls.align_2d_outer(
            [true_column_count, true_row_count], [pred_column_count, pred_row_count], transpose_rewards
        )

        cell_loc_map = {}
        for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
            for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
                cell_loc_map[(pred_row_num, pred_column_num)] = (true_row_num, true_column_num)

        return cell_loc_map


def _item_reward_function(pred_text, gold_text, pred_bbox, gold_bbox):
    def get_char_counter(s):
        if s == '':
            return Counter()
        return Counter(normalize_text(s))

    def compute_char_f1(pred_text, gold_text):
        pred_char_counter = get_char_counter(pred_text)
        gold_char_counter = get_char_counter(gold_text)
        intersection = gold_char_counter & pred_char_counter
        intersection = sum([int(value) for value in intersection.values()])
        extract_total = float(sum(value for value in pred_char_counter.values()))
        gold_total = float(sum(value for value in gold_char_counter.values()))

        if not pred_char_counter:
            precision = 1.0
        else:
            precision = intersection / extract_total
        if not gold_char_counter:
            recall = 1.0
        else:
            recall = intersection / gold_total
        f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
        return f1

    text_f1 = compute_char_f1(pred_text, gold_text)
    bbox_iou = overlap_iou(pred_bbox, gold_bbox)
    return text_f1 * 100 + 1e-9 * bbox_iou


def item_reward_function(pred_obj, gold_obj):
    pred_text = TextParser.to_string(pred_obj.get('answer', ''))
    gold_text = gold_obj['answer']

    pred_bbox = BboxParser.parse(pred_obj.get('bbox', []))
    gold_bbox = gold_obj['bbox']

    return _item_reward_function(pred_text, gold_text, pred_bbox, gold_bbox)


def compute_one(parsed_pred_text, parsed_gold_text, pred_bbox, gold_bbox):
    if parsed_pred_text is None or parsed_gold_text is None:
        em_score = f1_score = 0
    else:
        em_score = exact_match(parsed_pred_text, parsed_gold_text)
        f1_score = compute_f1(parsed_pred_text, parsed_gold_text)
    if pred_bbox is None or gold_bbox is None:
        iou_score = 0
    else:
        iou_score = overlap_iou(pred_bbox, gold_bbox)

    return em_score, f1_score, iou_score


def evaluate_one_pair(instance_id, task_name, sub_task_name, raw_question, pred_key, pred_obj, gold_key, gold_obj):
    def form_idx(key_item):
        if key_item is None:
            return key_item
        if isinstance(key_item, tuple):
            return ','.join([str(i) for i in key_item])
        return str(key_item)

    assert not (pred_obj is None and gold_obj is None), 'At least one valid obj'

    ori_pred_text = pred_bbox = None
    ori_gold_text = gold_bbox = None
    if pred_obj is not None:
        ori_pred_text = str(pred_obj.get('answer', ''))
        pred_bbox = pred_obj.get('bbox', [])
    if gold_obj is not None:
        ori_gold_text = str(gold_obj['answer'])
        gold_bbox = gold_obj['bbox']

    is_evidence = is_reasoning_task(task_name) and (gold_key is not None or pred_key is not None)
    parsed_pred_text, parsed_gold_text = TextParser.parse(
        ori_pred_text, ori_gold_text, raw_question, task_name, sub_task_name, is_evidence
    )
    parsed_pred_bbox = BboxParser.parse(pred_bbox)
    em_score, f1_score, iou_score = compute_one(parsed_pred_text, parsed_gold_text, parsed_pred_bbox, gold_bbox)

    if pred_obj is not None:
        pred_bbox = str(pred_bbox)
    if gold_obj is not None:
        gold_bbox = str(gold_bbox)

    result = {
        'index': instance_id,
        'pred_id': form_idx(pred_key),
        'gold_id': form_idx(gold_key),
        'ori_pred_text': ori_pred_text,
        'pred_text': parsed_pred_text,
        'gold_text': parsed_gold_text,
        'ori_pred_bbox': pred_bbox,
        'pred_bbox': parsed_pred_bbox,
        'gold_bbox': gold_bbox,
        'em': em_score,
        'f1': f1_score,
        'iou': iou_score,
        'iou@0.2': 1 if iou_score > 0.2 else 0,
        'iou@0.5': 1 if iou_score > 0.5 else 0,
        'iou@0.7': 1 if iou_score > 0.7 else 0,
        'ori_gold_text': ori_gold_text,
    }

    return result


def evaluate_details(
    instance_id, task_name, sub_task_name, raw_question, pred2gold_map, pred_results, gold_results, recall_only=False
):
    details = []
    for pred_key, gold_key in pred2gold_map.items():
        pred_obj = pred_results[pred_key]
        gold_obj = gold_results[gold_key]

        details.append(
            evaluate_one_pair(
                instance_id, task_name, sub_task_name, raw_question, pred_key, pred_obj, gold_key, gold_obj
            )
        )

    iterator = pred_results.items() if isinstance(pred_results, dict) else enumerate(pred_results)
    for pred_key, pred_obj in iterator:
        if pred_key in pred2gold_map or recall_only:
            continue
        details.append(
            evaluate_one_pair(instance_id, task_name, sub_task_name, raw_question, pred_key, pred_obj, None, None)
        )

    gold2pred_map = {v: k for k, v in pred2gold_map.items()}
    iterator = gold_results.items() if isinstance(gold_results, dict) else enumerate(gold_results)
    for gold_key, gold_obj in iterator:
        if gold_key in gold2pred_map:
            continue
        details.append(
            evaluate_one_pair(instance_id, task_name, sub_task_name, raw_question, None, None, gold_key, gold_obj)
        )

    return details


def compute_array_metric(instance_details, recall_only=False):
    def compute_metric(aligned_score, num_true, num_positives, num_invalid):
        selection = 0 if not recall_only else -1
        num_true = num_true - num_invalid
        num_positives = num_positives - num_invalid
        if not num_true and not num_positives:
            return None
        return Grits.compute_fscore(aligned_score, num_true, num_positives)[selection]

    aligned_em = aligned_f1 = aligned_iou = 0
    aligned_iou2 = aligned_iou5 = aligned_iou7 = 0
    num_true = num_positives = 0
    em_num_invalid = f1_num_invalid = iou_num_invalid = 0
    iou2_num_invalid = iou5_num_invalid = iou7_num_invalid = 0
    for one in instance_details:
        if one['pred_id'] is not None and one['gold_id'] is not None:
            if one['em'] is not None:
                aligned_em += one['em']
            else:
                em_num_invalid += 1
            if one['f1'] is not None:
                aligned_f1 += one['f1']
            else:
                f1_num_invalid += 1
            if one['iou'] is not None:
                aligned_iou += one['iou']
            else:
                iou_num_invalid += 1
            if one['iou@0.2'] is not None:
                aligned_iou2 += one['iou@0.2']
            else:
                iou2_num_invalid += 1
            if one['iou@0.5'] is not None:
                aligned_iou5 += one['iou@0.5']
            else:
                iou5_num_invalid += 1
            if one['iou@0.7'] is not None:
                aligned_iou7 += one['iou@0.7']
            else:
                iou7_num_invalid += 1
        if one['pred_id'] is not None:
            num_positives += 1
        if one['gold_id'] is not None:
            num_true += 1

    em = compute_metric(aligned_em, num_true, num_positives, em_num_invalid)
    f1 = compute_metric(aligned_f1, num_true, num_positives, f1_num_invalid)
    iou = compute_metric(aligned_iou, num_true, num_positives, iou_num_invalid)
    iou2 = compute_metric(aligned_iou2, num_true, num_positives, iou2_num_invalid)
    iou5 = compute_metric(aligned_iou5, num_true, num_positives, iou5_num_invalid)
    iou7 = compute_metric(aligned_iou7, num_true, num_positives, iou7_num_invalid)
    instance_id = instance_details[0]['index']
    metric_dict = {'index': instance_id, 'em': em, 'f1': f1, 'iou': iou, 'iou@0.2': iou2, 'iou@0.5': iou5, 'iou@0.7': iou7}
    return metric_dict


def compute_reasoning_metric(final_answer_metric, evidence_metric):
    if evidence_metric['em'] is None:
        em = final_answer_metric['em']
    else:
        em = (final_answer_metric['em'] + evidence_metric['em']) / 2

    if evidence_metric['f1'] is None:
        f1 = final_answer_metric['f1']
    else:
        f1 = (final_answer_metric['f1'] + evidence_metric['f1']) / 2

    if final_answer_metric['iou'] is None:
        iou = evidence_metric['iou']
    else:
        iou = (final_answer_metric['iou'] + evidence_metric['iou']) / 2

    if final_answer_metric['iou@0.2'] is None:
        iou2 = evidence_metric['iou@0.2']
    else:
        iou2 = (final_answer_metric['iou@0.2'] + evidence_metric['iou@0.2']) / 2

    if final_answer_metric['iou@0.5'] is None:
        iou5 = evidence_metric['iou@0.5']
    else:
        iou5 = (final_answer_metric['iou@0.5'] + evidence_metric['iou@0.5']) / 2

    if final_answer_metric['iou@0.7'] is None:
        iou7 = evidence_metric['iou@0.7']
    else:
        iou7 = (final_answer_metric['iou@0.7'] + evidence_metric['iou@0.7']) / 2

    return {'index': final_answer_metric['index'], 'em': em, 'f1': f1, 'iou': iou, 'iou@0.2': iou2, 'iou@0.5': iou5, 'iou@0.7': iou7}


def extract_raw_question(obj):
    if 'raw_question' in obj:
        raw_question = obj['raw_question']
    else:
        raw_question = re.search(r'# Task(.)+### Requirements', obj['question'], re.DOTALL).group().strip()
    return raw_question


def evaluate_table(task_name, data, model_name):
    def array2dict(array_input):
        result_dict = {}
        row_count = len(array_input)
        column_count = len(array_input[0]) if row_count else 0
        for row in range(row_count):
            for col in range(column_count):
                result_dict[(row, col)] = array_input[row][col] if col < len(array_input[row]) else {}
        return result_dict

    details = []
    metrics = []

    total = data.shape[0]
    for i in tqdm(range(total), desc=f'Evaluating {task_name}'):
        line = data.iloc[i]
        instance_id = line['index']
        task_name = line['task']
        sub_task_name = line['sub_task']
        raw_question = extract_raw_question(line)

        pred_2d_results = PredictionExtractor.process_prediction(
            str(line['prediction']),
            task_name=line['task'],
            default='2d_array',
            model_name=model_name,
            raw_question=raw_question,
            sub_task_name=line['sub_task']
        )
        gold_2d_results = json.loads(line['answer'])
        cell_loc_map = Grits.find_2d_alignment(gold_2d_results, pred_2d_results, item_reward_function)

        pred_2d_result_by_key = array2dict(pred_2d_results)
        gold_2d_results_by_key = array2dict(gold_2d_results)

        instance_details = evaluate_details(
            instance_id,
            task_name,
            sub_task_name,
            raw_question,
            cell_loc_map,
            pred_2d_result_by_key,
            gold_2d_results_by_key,
        )
        for d in instance_details:
            d['prediction_json'] = json.dumps(pred_2d_results)
        details.extend(instance_details)
        metric_dict = compute_array_metric(instance_details)
        metrics.append(metric_dict)

    return metrics, details


def evaluate_reasoning(task_name, data, model_name):
    details = []
    metrics = []
    total = data.shape[0]
    for i in tqdm(range(total), desc=f'Evaluating {task_name}'):
        line = data.iloc[i]

        pred_dict = PredictionExtractor.process_prediction(
            str(line['prediction']),
            task_name=line['task'],
            model_name=model_name,
            raw_question=line['raw_question'],
            sub_task_name=line['sub_task']
        )
        gold_dict = json.loads(line['answer'])[0]

        final_answer_detail = evaluate_one_pair(
            line['index'],
            line['task'],
            line['sub_task'],
            extract_raw_question(line),
            None,
            pred_dict,
            None,
            gold_dict,
        )
        if not gold_dict['bbox']:
            final_answer_detail['iou'] = None
            final_answer_detail['iou@0.2'] = None
            final_answer_detail['iou@0.5'] = None
            final_answer_detail['iou@0.7'] = None
        final_answer_detail['prediction_json'] = json.dumps(pred_dict)
        details.append(final_answer_detail)

        pred_evidence = pred_dict.get('supporting_evidence', [])
        gold_evidence = gold_dict['supporting_evidence']
        for e in gold_evidence:
            e['text'] = '' if e['text'] is None else e['text']

        if not gold_evidence:
            logger.warning(f'Invalid instance {line["index"]}')
            continue

        for e in pred_evidence + gold_evidence:
            e['answer'] = e.pop('text', '')

        pre_computed_rewards = HungarianMatcher.compute_1d_rewards(pred_evidence, gold_evidence, item_reward_function)
        seq_map = HungarianMatcher.align_1d(pre_computed_rewards)
        evidence_details = evaluate_details(
            line['index'],
            line['task'],
            line['sub_task'],
            extract_raw_question(line),
            seq_map,
            pred_evidence,
            gold_evidence,
            recall_only=True,
        )
        for item in evidence_details:
            item['prediction_json'] = json.dumps(pred_dict)
            if len(item['gold_text']) == 0:
                item['em'] = None
                item['f1'] = None
        evidence_metric_dict = compute_array_metric(evidence_details, recall_only=True)

        metrics.append(compute_reasoning_metric(final_answer_detail, evidence_metric_dict))
        details.extend(evidence_details)
    return metrics, details


def evaluate_simple(task_name, data, model_name):
    details = []
    metrics = []
    total = data.shape[0]
    for i in tqdm(range(total), desc=f'Evaluating {task_name}'):
        line = data.iloc[i]
        task_name = line['task']
        sub_task_name = line['sub_task']
        pred_dict = PredictionExtractor.process_prediction(
            str(line['prediction']),
            task_name=task_name,
            model_name=model_name,
            raw_question=line['raw_question'],
            sub_task_name=line['sub_task']
        )
        gold_dict = json.loads(line['answer'])[0]

        instance_detail = evaluate_one_pair(
            line['index'],
            line['task'],
            line['sub_task'],
            extract_raw_question(line),
            None,
            pred_dict,
            None,
            gold_dict,
        )

        if sub_task_name == 'Text2Bbox':
            instance_detail['em'] = None
            instance_detail['f1'] = None

        if sub_task_name == 'Bbox2Text':
            instance_detail['iou'] = None
            instance_detail['iou@0.2'] = None
            instance_detail['iou@0.5'] = None
            instance_detail['iou@0.7'] = None

        instance_detail['prediction_json'] = json.dumps(pred_dict)
        details.append(instance_detail)
        metrics.append({selected_key: instance_detail[selected_key] for selected_key in ['index', 'em', 'f1', 'iou', 'iou@0.2', 'iou@0.5', 'iou@0.7']})

    return metrics, details


def evaluate_list(task_name, data, model_name):
    details = []
    metrics = []
    total = data.shape[0]
    for i in tqdm(range(total), desc=f'Evaluating {task_name}'):
        line = data.iloc[i]

        pred_1d_results = PredictionExtractor.process_prediction(
            str(line['prediction']),
            task_name=line['task'],
            default='1d_array',
            model_name=model_name,
            raw_question=line['raw_question'],
            sub_task_name=line['sub_task']
        )
        gold_1d_results = json.loads(line['answer'])

        pre_computed_rewards = HungarianMatcher.compute_1d_rewards(
            pred_1d_results, gold_1d_results, item_reward_function
        )
        seq_map = HungarianMatcher.align_1d(pre_computed_rewards)

        instance_details = evaluate_details(
            line['index'],
            line['task'],
            line['sub_task'],
            extract_raw_question(line),
            seq_map,
            pred_1d_results,
            gold_1d_results,
        )
        for d in instance_details:
            d['prediction_json'] = json.dumps(pred_1d_results)
        details.extend(instance_details)
        metric_dict = compute_array_metric(instance_details)
        metrics.append(metric_dict)

    return metrics, details


def save_metrics(metrics, details, data, save_path):
    data = update_data_format(data)
    df_details = pd.DataFrame(details)
    df_details = pd.merge(df_details, data, how='left', on='index')
    df_instance_metrics = pd.DataFrame(metrics)
    df_instance_metrics = pd.merge(data, df_instance_metrics, how='left', on='index')
    df_sub_details = df_details[['index', 'prediction_json']]
    df_sub_details = df_sub_details.groupby('index').first().reset_index()
    df_instance_metrics = df_instance_metrics.merge(df_sub_details, how='left', on='index')

    paper_index = pd.MultiIndex.from_tuples(paper_order, names=['task', 'sub_task'])
    df_sub_task_metrics = (
        df_instance_metrics.groupby([CAPABILITY_NAME, 'task', 'sub_task'])[['em', 'f1', 'iou', 'iou@0.2', 'iou@0.5', 'iou@0.7']].mean().reset_index()
    )
    df_sub_task_metrics_indexed = df_sub_task_metrics.set_index(['task', 'sub_task'])
    df_sub_task_metrics = df_sub_task_metrics_indexed.reindex(paper_index).reset_index()
    df_task_metrics = df_sub_task_metrics.groupby([CAPABILITY_NAME, 'task'])[['em', 'f1', 'iou', 'iou@0.2', 'iou@0.5', 'iou@0.7']].mean().reset_index()
    df_task_metrics['task_categorical'] = pd.Categorical(df_task_metrics['task'], categories=task_order, ordered=True)
    df_task_metrics = df_task_metrics.sort_values(by='task_categorical').drop(columns='task_categorical')
    df_capability_metrics = df_task_metrics.groupby(CAPABILITY_NAME)[['em', 'f1', 'iou', 'iou@0.2', 'iou@0.5', 'iou@0.7']].mean().reset_index()
    df_instance_metrics = df_instance_metrics[
        ['index', CAPABILITY_NAME, 'task', 'sub_task', 'em', 'f1', 'iou', 'iou@0.2', 'iou@0.5', 'iou@0.7', 'prediction', 'prediction_json', 'answer']
    ]
    df_overall_metrics = df_capability_metrics[['em', 'f1', 'iou', 'iou@0.2', 'iou@0.5', 'iou@0.7']].mean().to_frame().T

    with pd.ExcelWriter(
        save_path,
        engine='xlsxwriter',
        engine_kwargs={'options': {'strings_to_formulas': False, 'strings_to_urls': False}},
    ) as writer:
        df_overall_metrics.to_excel(writer, sheet_name='Overall', index=False)
        df_capability_metrics.to_excel(writer, sheet_name='Category Level', index=False)
        df_task_metrics.to_excel(writer, sheet_name='Main Task Level', index=False)
        df_sub_task_metrics.to_excel(writer, sheet_name='Sub Task(Dataset) Level', index=False)
        df_instance_metrics.to_excel(writer, sheet_name='Instance Level', index=False)
        df_details.to_excel(writer, sheet_name='Details', index=False)

    df_overall_metrics[CAPABILITY_NAME] = 'Overall'
    df_score = pd.concat([df_capability_metrics, df_overall_metrics], ignore_index=True)
    df_score = df_score.melt(id_vars=[CAPABILITY_NAME], var_name='Score Name', value_name='Value')
    df_score['Score Name'] = df_score[CAPABILITY_NAME] + ' ' + df_score['Score Name'].str.upper()
    df_score['Value'] = df_score['Value'].round(4) * 100
    df_score = df_score[['Score Name', 'Value']].sort_values('Score Name', ignore_index=True)
    return df_score


def update_data_format(df_data):
    df_data = df_data.rename(columns={'capability': CAPABILITY_NAME})
    if 'dataset_name' in df_data.columns:
        df_data['sub_task'] = df_data['dataset_name']
        df_data.drop(columns=['dataset_name'], inplace=True)
        df_data.loc[df_data['sub_task'] == 'OCR-VQA', 'sub_task'] = 'BookOCR'
        df_data.loc[df_data[CAPABILITY_NAME] != 'Visual Reasoning', CAPABILITY_NAME] = 'Visual Perception'
        df_data['reasoning_type'] = df_data['reasoning_type'].fillna('')
        df_data.loc[df_data['reasoning_type'] == 'algebraic', 'reasoning_type'] = 'arithmetic'
        df_data.loc[df_data['task'] == 'Text Localization Bbox2Text', 'sub_task'] = 'Bbox2Text'
        df_data.loc[df_data['task'] == 'Text Localization Text2Bbox', 'sub_task'] = 'Text2Bbox'
        df_data.loc[df_data['task'] == 'Text Localization Bbox2Text', 'task'] = 'Text Localization'
        df_data.loc[df_data['task'] == 'Text Localization Text2Bbox', 'task'] = 'Text Localization'

        df_perception = df_data[df_data[CAPABILITY_NAME] == 'Visual Perception']
        df_reasoning = df_data[df_data[CAPABILITY_NAME] == 'Visual Reasoning'].copy()
        df_reasoning['task'] = df_reasoning['reasoning_type'].str.title()
        df_reasoning['task'] = df_reasoning['task'].apply(
            lambda x: x + ' Reasoning' if x in ['Arithmetic', 'Logical', 'Spatial'] else x
        )
        df_reasoning = df_reasoning.drop(columns=['reasoning_type'])  # mute SettingWithCopyWarning warnings
        return pd.concat([df_perception, df_reasoning], ignore_index=True)
    return df_data


def evaluate(data, model_name):
    data = update_data_format(data)
    details = []
    metrics = []
    grouped_data = data.groupby('task')
    for task_name, task_data in grouped_data:
        if is_reasoning_task(task_name):
            instance_metrics, task_details = evaluate_reasoning(task_name, task_data, model_name)
        elif task_name == 'Table Recognition':
            instance_metrics, task_details = evaluate_table(task_name, task_data, model_name)
        elif task_name == 'Document Forgery Detection':
            instance_metrics, task_details = evaluate_list(task_name, task_data, model_name)
        elif task_name == 'Text Recognition':
            inner_grouped_data = task_data.groupby('sub_task')
            task_details = []
            instance_metrics = []
            for sub_task, dataset_data in inner_grouped_data:
                if sub_task == 'BookOCR':
                    dataset_instance_metrics, dataset_details = evaluate_list(
                        f'{task_name} {sub_task}', dataset_data, model_name
                    )
                else:
                    dataset_instance_metrics, dataset_details = evaluate_simple(
                        f'{task_name} {sub_task}', dataset_data, model_name
                    )
                task_details.extend(dataset_details)
                instance_metrics.extend(dataset_instance_metrics)
        else:
            instance_metrics, task_details = evaluate_simple(task_name, task_data, model_name)

        details.extend(task_details)
        metrics.extend(instance_metrics)

    return metrics, details


def load_excels(file_list):
    input_dfs = []
    for one_file in file_list:
        input_dfs.append(pd.read_excel(one_file))
    df_concatenated = pd.concat(input_dfs)
    return df_concatenated
