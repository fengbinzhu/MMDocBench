import math
import re
import string
from dataclasses import dataclass
from typing import Optional

import regex

from vlmeval.smp.log import get_logger

logger = get_logger('MMDocBench Evaluation')


def digit_fix(text):
    manualMap = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    ori_text = text
    for string_number, digit_number in manualMap.items():
        text = re.sub(rf'\b{string_number}\b', digit_number, text, re.I)

    if ori_text != text:
        logger.debug(f'[Digit Fix] {repr(ori_text)} -----> {repr(text)}')

    return text


def normalize_text(s, keep_newline=False):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        _regex = regex.compile(r'\b(a|an|the)\b', regex.UNICODE)
        return regex.sub(_regex, ' ', text)

    def white_space_fix(text):
        if keep_newline:
            text = text.replace('\n', '⚽')
            text = ' '.join(text.split())
            text = text.replace('⚽', '\n')
            return text
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(digit_fix(remove_articles(remove_punc(lower(s)))))


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values


def is_reasoning_task(task_name):
    return task_name in [
        'Arithmetic Reasoning',
        'Logical Reasoning',
        'Spatial Reasoning',
        'Comparison',
        'Sorting',
        'Counting',
    ]


@dataclass
class NumberValue:
    parsed_value: str
    status: bool
    from_percentage: bool = False
    decimal_num: Optional[int] = None

    def restore_to_value_before_percentage(self):
        self.parsed_value = str(float(self.parsed_value) * 100)
        self.decimal_num = max(0, self.decimal_num - 2)


class TextParser:
    @staticmethod
    def parse_number(value):
        if re.search(r'[,]\s?\d{2}\D*$', value):
            idx = value.rfind(r',')
            if idx != -1:
                value = value[:idx] + '.' + value[idx + 1:]
        percentage_flag = False
        if re.search(r'(\s[-+]?[\d.]*\d+%)|(^[-+]?[\d.]*\d+%)', value):
            percentage_flag = True
        # remove disturbing '-'
        value = re.sub(r'(?<=[a-z] )-(?= \d+)', '', str(value), flags=re.I)
        s = ''.join([ch for ch in str(value) if ch not in '\'"\\$€£¥%(),[]* '])
        r_num = r'([+-]?\d+(\.\d+)?)|([+-]?\.\d+)'
        groups = re.findall(r_num, s)
        if len(groups) == 0:
            return NumberValue(value, False, False, None)
        num = groups[0][0] or groups[0][2]
        if num == '':
            return NumberValue(value, False, False, None)
        if '.' in num:
            if percentage_flag:
                float_value = str(float(num) / 100)
            else:
                float_value = str(float(num))
            decimal_num = len(float_value) - float_value.rfind('.') - 1
            return NumberValue(float_value, status=True, from_percentage=percentage_flag, decimal_num=decimal_num)
        elif percentage_flag:
            float_value = str(float(num) / 100)
            decimal_num = len(float_value) - float_value.rfind('.') - 1
            return NumberValue(float_value, status=True, from_percentage=percentage_flag, decimal_num=decimal_num)
        return NumberValue(str(int(num)), status=True, from_percentage=percentage_flag, decimal_num=None)

    @staticmethod
    def extract_from_equation(pred_text, raw_question):
        equation_m = re.search(
            (
                r'-?[$]?\d[\d,.]*(?:%| million)?(?:(?:\s*[+\-*/]\s*-?[$]?\d[\d,.]*(?:%| million)?)'
                r'+\s*=\s*(-?[$]?\d[\d,.]*(?:%| million)?))+'
            ),
            pred_text,
        )
        if equation_m:
            return equation_m.group(1).strip('.')
        return None

    @staticmethod
    def extract_number_from_by(pred_text, raw_question):
        if re.search('by how (?:much|many)', raw_question, re.I):
            targets = [t for t in re.split(r'\bby\b', pred_text, re.I)[1:] if re.search(r'\d', t)]
            if targets:
                return targets[-1].strip('.')
        return None

    @staticmethod
    def extract_from_predicate(pred_text, raw_question):
        predicates = re.findall(r'\b(?:is|be|was)\b', pred_text)
        if len(predicates) <= 1:
            targets = [t for t in re.split(r'\b(?:is|be|was)\b', pred_text)[1:] if re.search(r'\d', t)]
            if targets:
                return targets[-1]  # parse_number can be used safely
        else:
            indicator_m = re.search(r'\b(?:therefore|which(?=(?: is|would be|was)))\b', pred_text, re.I)
            if indicator_m:
                interested_pred_text = pred_text[indicator_m.end():]
                predicates = re.findall(r'\b(?:is|be|was)\b', interested_pred_text)
                if len(predicates) <= 1:
                    interested_pred_text = re.split(r'\b(?:is|be|was)\b', interested_pred_text)[-1]
                    if re.search(r'\d', interested_pred_text):
                        return interested_pred_text  # parse_number can be used safely
            # remove noises
            pred_text = re.sub(r'bounding box[^.]*?is', '', pred_text, re.I)
            # split into candidates
            targets = re.split(r'\bis\b', pred_text)
            targets = [t for t in targets if re.search(r'^[^\d]{,20}[$()]*\d[\d,. ]*[$()]*', t)]
            if len(targets):
                return targets[-1]
        return None

    @staticmethod
    def extract_from_date_value_pair(pred_text, raw_question):
        m = re.search(r'(\d{4}:) (\$)?\d', pred_text)
        if m:
            return pred_text[m.end(1):]
        return None

    @classmethod
    def extract_number_part(cls, pred_text, raw_question=''):
        pred_text = digit_fix(pred_text)
        trials = [
            cls.extract_number_from_by,
            cls.extract_from_equation,
            cls.extract_from_predicate,
            cls.extract_from_date_value_pair,
        ]
        for extract_fn in trials:
            pred_number = extract_fn(pred_text, raw_question)
            if pred_number is not None:
                return pred_number

        number_parts = re.findall(r'\b\d[\d,. ]*\b', pred_text)
        if len(pred_text) and len(number_parts) != 1:
            logger.debug(f'Fail to extract number from {repr(pred_text)}. Returning ori value')
        return pred_text

    @staticmethod
    def is_gold_number(gold_text):
        m = re.search(
            (
                r'^(-|rp|\$|£)?[()\d,. %]+(pts|million|billion|articles?|ps|years?'
                r'|months?|weeks?|days?|earth days?|cm|mm|m|km|deaths?|experts?|'
                r'seconds?|minutes?|hours?|hours ahead|events?|bps|kg|g|gallons?'
                r'|percent|inches|votes?|k)?$'
            ),
            gold_text,
            re.I,
        )
        return m is not None

    @staticmethod
    def is_gold_yes(gold_text):
        return re.search(r'^yes\b', gold_text, re.I) is not None

    @staticmethod
    def is_pred_yes(pred_text):
        with_yes = re.search(r'^yes\b', pred_text, re.I) is not None
        without_no = re.search(r'^no\b', pred_text, re.I) is None
        no_not = len(pred_text) > 0 and without_no and re.search(r'\bnot\b', pred_text, re.I) is None
        return no_not or with_yes

    @staticmethod
    def is_gold_no(gold_text):
        return gold_text.lower() == 'no'

    @staticmethod
    def is_pred_no(pred_text):
        return re.search(r'^no\b|\bnot\b', pred_text, re.I) is not None

    @staticmethod
    def to_string(pred_text):
        if pred_text is None:
            return ''
        return str(pred_text)

    @staticmethod
    def remove_repetitions(s):
        # meaningless point listing
        points = [int(point) for point in re.findall(r'\b(\d+)\. (?=[a-z])', s, re.I)]
        if points and len(points) > 10 and points == list(range(points[0], points[-1] + 1)):
            logger.debug(f'[Repetition] remove invalid prediction completely: {repr(s[:100] + "...")}')
            return ''

        # general repetition removal
        r = re.compile(r'(.+?)\1+')
        detections = []
        for match in r.finditer(s):
            if match.end() - match.end(1) > 30:
                detections.append((match.group(1), match.end(1), match.end()))
        for r in detections:
            logger.debug(f'[Repetition] remove {r[0]} from string')
            s = s[: r[1]] + s[r[2]:]

        return s

    @classmethod
    def parse(cls, pred_text, gold_text, raw_question='', task_name='', sub_task_name='', is_evidence=False):
        pred_text = cls.to_string(pred_text)
        gold_text = cls.to_string(gold_text)

        pred_text = cls.remove_repetitions(pred_text)

        skip_parse = re.search(r'Recognition|Localization|Forgery', task_name) is not None
        keep_newline = 'Recognition' in task_name
        skip_normalize = False
        if not skip_parse and pred_text is not None and gold_text is not None:
            # 1. float/int only
            # 2. percentage
            # 3. number + units
            # 4. currency + number
            if cls.is_gold_number(gold_text):
                possible_negative = 'decrease' in pred_text
                interested_pred_text = cls.extract_number_part(pred_text, raw_question)
                pred_num = cls.parse_number(interested_pred_text)
                gold_num = cls.parse_number(gold_text)

                question_for_percentage = re.search(r'[a-z ]%[a-z ]|\bpercentage\b', raw_question, re.I)
                if pred_num.from_percentage and not gold_num.from_percentage and question_for_percentage:
                    pred_num.restore_to_value_before_percentage()

                pred_text = pred_num.parsed_value
                gold_text = gold_num.parsed_value

                if '.' not in gold_text and re.search(r'\.0+$', pred_text):
                    pred_text = str(int(float(pred_text)))
                elif pred_num.decimal_num and gold_num.decimal_num and pred_num.decimal_num != gold_num.decimal_num:
                    round_num = min(pred_num.decimal_num, gold_num.decimal_num)
                    pred_text = str(round(float(pred_text), round_num))
                    gold_text = str(round(float(gold_text), round_num))
                # ignore sign error in decrease setting
                if gold_text.startswith('-') and gold_text[1:] == pred_text and possible_negative:
                    pred_text = f'-{pred_text}'
                # ignore sign error in subtraction questions
                if (
                    pred_num.status
                    and not is_evidence
                    and 'difference' in raw_question
                    and 'absolute' not in raw_question
                    and gold_text != pred_text
                    and abs(float(gold_text)) == abs(float(pred_text))
                ):
                    pred_text = gold_text
                skip_normalize = True
            elif cls.is_gold_yes(gold_text) and cls.is_pred_yes(pred_text):
                pred_text = gold_text
            elif cls.is_gold_no(gold_text) and cls.is_pred_no(pred_text):
                pred_text = gold_text
            # long answer with disturbing context information
            elif 'Question Answering' in task_name or (is_reasoning_task(task_name)):
                if anls_compute(gold_text, pred_text) < 0.5 or gold_text.lower() in pred_text.lower():
                    pred_text = gold_text
        if pred_text is not None and not skip_normalize:
            pred_text = normalize_text(pred_text, keep_newline=keep_newline)
        if gold_text is not None and not skip_normalize:
            gold_text = normalize_text(gold_text, keep_newline=keep_newline)
        return pred_text, gold_text


class BboxParser:
    @classmethod
    def parse(cls, bbox, model_name=''):
        if bbox and isinstance(bbox, list):
            if len(bbox) == 4 and (isinstance(bbox[0], int) or isinstance(bbox[0], float)):
                if all(co < 2 for co in bbox):
                    x0 = math.floor(bbox[0] * 1000)
                    y0 = math.floor(bbox[1] * 1000)
                    x1 = math.ceil(bbox[2] * 1000)
                    y1 = math.ceil(bbox[3] * 1000)
                    return [x0, y0, x1, y1]
                return bbox
            elif isinstance(bbox[0], dict) and bbox[0].get('bbox'):
                return cls.parse(bbox[0]['bbox'])
            elif isinstance(bbox[0], list):
                return cls.parse(bbox[0])
            elif (
                isinstance(bbox[0], dict)
                and bbox[0].keys() == {'x0', 'y0'}
                and isinstance(bbox[1], dict)
                and bbox[1].keys() == {'x1', 'y1'}
            ):
                return list(bbox[0].values()) + list(bbox[1].values())
        logger.debug(f'Fail to parse bbox: {bbox}. Return empty')
        return []
