import math
import os
import re
import string
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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
        text = re.sub(rf'\b{string_number}\b', digit_number, text, flags=re.I)

    if ori_text != text:
        logger.debug(f'[Digit Fix] {repr(ori_text)} -----> {repr(text)}')

    return text

def month_fix(text):
    month_abbreviations = {
        'jan': 'January',
        'feb': 'February',
        'mar': 'March',
        'apr': 'April',
        'may': 'May',  # May is the same abbreviation and full form
        'jun': 'June',
        'jul': 'July',
        'aug': 'August',
        'sep': 'September',
        'oct': 'October',
        'nov': 'November',
        'dec': 'December',
    }
    # Create a regex pattern that matches any of the abbreviations.
    # The r'\b' ensures that we match whole words (word boundaries).
    pattern = r'\b(' + '|'.join(month_abbreviations.keys()) + r')(?:\b|(?=\d))'

    def replace_match(match):
        abbr = match.group(0)
        return month_abbreviations[abbr.lower()]

    return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)


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


def normalize_narrative_text(text: str, raw_question: str, sub_task_name: str) -> str:
    if re.search(r'\bdate\b', raw_question, re.I):
        # replace month abbreviations with full value, e.g., Dec -> December
        text = month_fix(text)

    # Remove diacritics
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    text = re.sub(r'[‘’´`]', "'", text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‐‑‒–—−]', '-', text)
    while sub_task_name == 'WTQ':
        old_x = text
        # Remove citations
        text = re.sub(r'((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$', '', text.strip())
        # Remove details in parentheses
        text = re.sub(r'(?<!^)( \([^)]*\))*$', '', text.strip())
        # Remove outermost quotation mark
        text = re.sub(r'^"([^"]*)"$', r'\1', text.strip())
        if text == old_x:
            break
    return text


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

def is_narrative_kie(task_name: str, raw_question: str):
    narrative_tags = [
        'item name',
        'add-on name',
        'date',
        'address',
        'company',
        'Store_addr_value',
        'Store_name_value',
        'Tel_value',
        'Date_value',
        'Time_value'
    ]
    if task_name == 'Key Information Extraction':
        return re.search(fr'"(?:{"|".join(narrative_tags)})"', raw_question, re.I)

    return False


@dataclass
class NumberValue:
    parsed_value: str
    status: bool
    from_percentage: bool = False
    decimal_num: Optional[int] = None

    def __post_init__(self):
        self.decimal_num = self.decimal_num or 0

    def restore_to_value_before_percentage(self):
        self.decimal_num = max(0, self.decimal_num - 2)
        if self.decimal_num == 0:
            self.parsed_value = str(int(Decimal(self.parsed_value) * 100))
        else:
            self.parsed_value = str(float(Decimal(self.parsed_value) * 100))
        self.from_percentage = False

    def get_rounded_str(self, round_num: int = 2):
        # make sure ('1.23', 3) -> '1.23'
        try:
            return str(float(round(Decimal(self.parsed_value), round_num)))
        except InvalidOperation:
            logger.warning(f'Invalid value: {self.parsed_value}')
            return self.parsed_value

    def get_int_str(self):
        return str(int(float(self.parsed_value)))

    def maybe_percentage_error(self, other_num: 'NumberValue'):
        if self.status and other_num.status and Decimal(self.parsed_value) * 100 == Decimal(other_num.parsed_value):
            return True
        return False

    def maybe_rounding_error(self, other_num: 'NumberValue'):
        min_decimal_num = min(self.decimal_num, other_num.decimal_num)
        diff_from_other = abs(Decimal(self.parsed_value) - Decimal(other_num.parsed_value))
        tolerance = (Decimal('0.1') ** min_decimal_num) * Decimal('0.1')
        if diff_from_other <= tolerance:
            return True
        return False

    def maybe_sign_error(self, other_num: 'NumberValue'):
        ori_unequal = self.parsed_value != other_num.parsed_value
        if other_num.status and ori_unequal and abs(Decimal(self.parsed_value)) == abs(Decimal(other_num.parsed_value)):
            return True
        return False

    def maybe_thousands_error(self, other_num: 'NumberValue'):
        if self.status and other_num.status and Decimal(self.parsed_value) * 1000 == Decimal(other_num.parsed_value):
            return True


class TextParser:
    @staticmethod
    def parse_number(value):
        if re.search(r',\s?\d{2}\D*$', value):
            idx = value.rfind(r',')
            if idx != -1:
                value = value[:idx] + '.' + value[idx + 1:]
        percentage_flag = False
        percentage_exists = re.search(r'(\s[-+]?[\d.]*\d+ ?%)|(^[-+]?[\d.]*\d+ ?%)', value)
        not_alternative = re.search(
            r'(?:\d{1,3}(?:,\d{3})*|\d+\.\d+ million,)\s+or\s+\d{1,2}(?:\.\d+)? ?%', value
        ) is None
        if percentage_exists and not_alternative:
            percentage_flag = True
        # remove disturbing '-'
        value = re.sub(r'(?<=[a-z] )-(?= [$€£¥]?\d+)', '', str(value), flags=re.I)
        # remove disturbing '.'
        value = re.sub(r'\([a-z,. ]+\)', '', str(value), flags=re.I)
        # remove Rp.
        value = re.sub(r'\brp\.', '', str(value), flags=re.I)
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
                float_value = str(float(Decimal(num) / 100))
            else:
                float_value = str(float(num))
            decimal_num = len(float_value) - float_value.rfind('.') - 1
            return NumberValue(float_value, status=True, from_percentage=percentage_flag, decimal_num=decimal_num)
        elif percentage_flag:
            float_value = str(float(Decimal(num) / 100))
            decimal_num = len(float_value) - float_value.rfind('.') - 1
            return NumberValue(float_value, status=True, from_percentage=percentage_flag, decimal_num=decimal_num)
        return NumberValue(str(int(num)), status=True, from_percentage=percentage_flag, decimal_num=None)

    @staticmethod
    def parse_ordinal_number(value):
        pattern = r'(?:^|(?<=\s))(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:$|(?=\s))'
        m = re.search(pattern, value, re.I)
        if m:
            return m.group()
        if re.search(r'(?:^|\s)1st(?:$|\s)', value, re.I):
            return 'first'
        if re.search(r'(?:^|\s)2nd(?:$|\s)', value, re.I):
            return 'second'
        if re.search(r'(?:^|\s)3rd(?:$|\s)', value, re.I):
            return 'third'
        if re.search(r'(?:^|\s)\dth(?:$|\s)', value, re.I):
            idx = int(re.search(r'(?:^|\s)(\d)th(?:$|\s)', value, re.I).group(1)) - 4
            return ['fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'][idx]
        return None

    @staticmethod
    def parse_date(value):
        value = month_fix(value)
        month_map = {
            'january': 1,
            'february': 2,
            'march': 3,
            'april': 4,
            'may': 5,
            'june': 6,
            'july': 7,
            'august': 8,
            'september': 9,
            'october': 10,
            'november': 11,
            'december': 12
        }
        month_pattern = r'january|february|march|april|may|june|july|august|september|october|november|december'
        m = re.search(rf"({month_pattern})(\d{{1,2}})'(\d{{2}})", value, re.I)
        if m:
            year = int(m.group(3))
            month = month_map[m.group(1).lower()]
            day = int(m.group(2))
            return year, month, day

        m = re.search(rf"\b(\d{{2}})\s*({month_pattern})\s*(\d{{4}})\b",  value, re.I)
        if m:
            month = month_map[m.group(2).lower()]
            day = int(m.group(1))
            year = int(m.group(3))
            return year, month, day

        m = re.search(rf"\b(\d{{2}})\s*({month_pattern})\s*(\d{{2}})\b", value, re.I)
        if m:
            # guess
            month =  month_map[m.group(2).lower()]
            day = int(m.group(1))
            year = int(m.group(3))
            return year, month, day

        m = re.search(rf'({month_pattern})\s+(\d{{1,2}}),\s*(\d{{4}})\b', value, re.I)
        if m:
            month = month_map[m.group(1).lower()]
            day = int(m.group(2))
            year = int(m.group(3))
            return year, month, day

        m = re.search(r'\b(\d{1,4})[-/](\d{1,4})[-/](\d{1,4})\b', value, re.I)
        if m:
            # guess
            if len(m.group(1)) == 4 and int(m.group(2)) <= 12:
                result_tuple = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif len(m.group(1)) == 1 and int(m.group(2)) > 12:
                result_tuple = (int(m.group(3)), int(m.group(1)), int(m.group(2)))
            elif len(m.group(1)) == 1:
                result_tuple = (int(m.group(3)), int(m.group(2)), int(m.group(1)))
            else:
                result_tuple = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return result_tuple


    @classmethod
    def number_in(cls, pred_text, gold_text):
        """
        Determines if the `gold_text` is mentioned within `pred_text`.
        """
        if len(gold_text) >= 3 or re.search(r'^\d+(?:, \d+){4,}$', pred_text):
            if re.search(fr'(?:^| |\b){re.escape(gold_text)}\.?(?:\b(?![,.]\d)| |$)', pred_text, re.I):
                return True
            number_parts = re.findall(r'\b\d[\d,. ]*\b', pred_text)
            number_parts = [pred_text] + number_parts
            for one_part in number_parts:
                pred_num = cls.parse_number(one_part)
                gold_num = cls.parse_number(gold_text)
                if pred_num.status and gold_num.status:
                    return pred_num.parsed_value == gold_num.parsed_value
        return False

    @staticmethod
    def extract_from_equation(pred_text, gold_text, raw_question, is_evidence):
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
    def extract_number_from_by(pred_text, gold_text, raw_question, is_evidence):
        if re.search('(?:by how (?:much|many))|(?:how much .+ change by)', raw_question, re.I):
            targets = [t for t in re.split(r'\bby\b', pred_text, re.I)[1:] if re.search(r'\d', t)]
            if targets:
                return targets[-1].strip('.')
        return None

    @classmethod
    def extract_from_predicate(cls, pred_text, gold_text, raw_question, is_evidence):
        predicates = re.findall(r'\b(?:is|be|was|are)\b', pred_text)
        if len(predicates) <= 1:
            targets = [t for t in re.split(r'\b(?:is|be|was|are)\b', pred_text)[1:] if re.search(r'\d', t)]
            if targets:
                return targets[-1]  # cls.parse_number can be used safely later
        else:
            # return gold text if it is in
            targets = [t for t in re.split(r'\b(?:is|be|was|are)\b', pred_text)[1:] if re.search(r'\d', t)]
            for one_target in targets:
                if cls.number_in(one_target, gold_text):
                    return gold_text

            # return the part where the prediction number is likely in
            indicator_m = re.search(r'\b(?:therefore|which(?= (?:is|would be|was|are)))\b', pred_text, re.I)
            if indicator_m:
                interested_pred_text = pred_text[indicator_m.end():]
                predicates = re.findall(r'\b(?:is|be|was|are)\b', interested_pred_text)
                if len(predicates) <= 1:
                    interested_pred_text = re.split(r'\b(?:is|be|was|are)\b', interested_pred_text)[-1]
                    if re.search(r'\d', interested_pred_text):
                        return interested_pred_text  # parse_number can be used safely
            # remove noises
            pred_text = re.sub(r'bounding box[^.]*?is', '', pred_text, re.I)
            # split into candidates
            targets = re.split(r'\b(?:is|be|was|are)\b', pred_text)
            targets = [t for t in targets if re.search(r'^\D{,20}[$()]*\d[\d,. ]*[$()]*', t)]
            if len(targets):
                return targets[-1]
        return None

    @staticmethod
    def extract_from_date_value_pair(pred_text, gold_text, raw_question, is_evidence):
        m = re.search(r'(\d{4}:) (\$)?\d', pred_text)
        if m:
            return pred_text[m.end(1):]

        if is_evidence and re.search(r'\byears?\b', raw_question, re.I):
            m = re.search(r'\d{4}\s+(\d[\d ,.]*)', pred_text)
            if m:
                return m.group(1)
        m = re.search(r'^\d{4} [a-z ]+: ?(\$?\d+)$', pred_text)
        if m:
            return m.group(1)

        return None

    @staticmethod
    def extract_from_year(pred_text, gold_text, raw_question, is_evidence):
        if 'which year' in raw_question and not is_evidence:
            number_parts = re.findall(r'\b\d{4}\b', pred_text)
            if len(number_parts) == 1:
                return number_parts[0]

        return None

    @staticmethod
    def extract_from_synonym(pred_text, gold_text, raw_question, is_evidence):
        if re.search(r'what was the change', raw_question, re.I) and gold_text == '0':
           if re.search(r'no change', pred_text, re.I):
               return gold_text
        elif re.search(r'^\d+ million$', gold_text, re.I) and re.search(r'^\d+,000,000$', pred_text, re.I):
            gold_num = int(re.search(r'^(\d+) million$', gold_text, re.I).group(1)) * 1000000
            pred_num = int(pred_text.replace(',', ''))
            if gold_num == pred_num:
                return gold_text

        return None

    @classmethod
    def extract_number_part(cls, pred_text, gold_text, raw_question='', is_evidence=False):
        pred_text = digit_fix(pred_text)
        trials = [
            cls.extract_from_synonym,
            cls.extract_number_from_by,
            cls.extract_from_equation,
            cls.extract_from_predicate,
            cls.extract_from_date_value_pair,
            cls.extract_from_year,
        ]
        for extract_fn in trials:
            pred_number = extract_fn(pred_text, gold_text, raw_question, is_evidence)
            if pred_number is not None:
                return pred_number

        number_parts = re.findall(r'\d[\d,. ]*', pred_text)
        if len(pred_text) and len(number_parts) != 1:
            if 'tax amount' in raw_question:
                valid_number_parts = [
                    np for np in pred_text.split() if re.search(r'\d', np) and not re.search(r'[a-zA-Z%]', np)
                ]
                if len(valid_number_parts) == 1:
                    return valid_number_parts[0]
            # if multiple numbers, choose the one with overlap
            if is_evidence:
                if cls.number_in(pred_text, gold_text):
                    return gold_text

            logger.debug(f'Fail to extract number from {repr(pred_text)}. Returning ori value')
        elif len(number_parts) == 1:
            # pred: 1,422,313 1,437,708 304,413 3,164,434
            # gold: 3,164,434
            if len(number_parts[0]) > len(gold_text):
                if cls.number_in(pred_text, gold_text):
                    return gold_text
        return pred_text

    @staticmethod
    def is_gold_number(gold_text):
        m = re.search(
            (
                r'^(-|rp|\$|£|€|eur)?[()\d,. %]+(pts|million|billion|bil\.|bn|articles?|ps|earth years|years?'
                r'|months?|weeks?|days?|earth days?|cm|mm|m|km|deaths?|experts?|'
                r'seconds?|minutes?|hours?|hrs|hours ahead|events?|bps|kg|g|gallons?'
                r'|percent|inches|votes?|k|b\.c\.?|a\.c\.?)?$'
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
        without_no = re.search(r'^no\b|\bnot\b', pred_text, re.I) is None
        no_not = len(pred_text) > 0 and without_no
        return no_not or with_yes

    @staticmethod
    def is_no_absolutely(gold_text):
        return gold_text.lower() == 'no'

    @staticmethod
    def is_no_guessing(pred_text):
        return re.search(r'^no\b|\bnot\b', pred_text, re.I) is not None

    @staticmethod
    def to_string(pred_text):
        if pred_text is None:
            return ''
        return str(pred_text)

    @classmethod
    def is_same_judgement(cls, pred_text, gold_text):
        if cls.is_gold_yes(gold_text) and cls.is_pred_yes(pred_text):
            return True
        elif cls.is_no_absolutely(gold_text) and cls.is_no_guessing(pred_text):
            return True
        elif cls.is_no_absolutely(pred_text) and cls.is_no_guessing(gold_text):
            return True

        return False


    @classmethod
    def is_same_ordinal_number(cls, pred_text, gold_text, raw_question):
        if re.search('which (?:one|amendment)', raw_question, re.I):
            pred_ordinal_number = cls.parse_ordinal_number(pred_text)
            gold_ordinal_number = cls.parse_ordinal_number(gold_text)
            if pred_ordinal_number is not None and gold_ordinal_number is not None:
                return pred_ordinal_number == gold_ordinal_number
        return False

    @classmethod
    def is_same_comparative_adjective(cls, pred_text, gold_text, raw_question):
        if re.search('more or less|greater than or less than', raw_question, re.I):
            pos_pred_adj = re.search(r'\b(?:more|greater)\b', pred_text, re.I)
            neg_pred_adj = re.search(r'\b(?:less|lower)\b', pred_text, re.I)
            pos_gold_adj = re.search(r'\b(?:more|greater)\b', gold_text, re.I)
            neg_gold_adj = re.search(r'\b(?:less|lower)\b', gold_text, re.I)
            if pos_pred_adj is not None and pos_gold_adj is not None:
                return True
            elif neg_pred_adj is not None and neg_gold_adj is not None:
                return True

        return False

    @classmethod
    def is_same_date(cls, pred_text, gold_text, raw_question):
        date_question = re.search('"(?:date_value|date)"', raw_question, re.I)
        if date_question is not None:

            # check if string sub-match
            pattern = r'date ?|mon ?|tue ?|wed ?|thu ?|fri ?|sat ?|sun ?'
            cleaned_gold_text = re.sub(pattern, '', gold_text, flags=re.I).replace(' ', '').lower()
            cleaned_pred_text = re.sub(pattern, '', pred_text, flags=re.I).replace(' ', '').lower()
            if cleaned_gold_text in cleaned_pred_text:
                return True

            # check if parsed match
            gold_tuple = cls.parse_date(gold_text)
            pred_tuple = cls.parse_date(pred_text)
            if gold_tuple is not None and pred_tuple is not None:
                if gold_tuple[1:] == pred_tuple[1:] and abs(gold_tuple[0] - pred_tuple[0]) in [0, 2000]:
                    return True
                elif gold_tuple == pred_tuple[::-1]:
                    return True

        return False

    @staticmethod
    def is_same_trend(pred_text, gold_text, raw_question, task_name, sub_task_name):
        if sub_task_name == 'CharXiv' and task_name == 'Logical Reasoning':
            if re.search(r'\b(?:relationship|vary with|change as)\b', raw_question, re.I):
                if re.search('^increase', gold_text, re.I):
                    straight_answer_pattern = r'(?:^increase)'
                    # as xxx increases, the xxx increases.
                    front_pattern = r'(?:as .+(?:increase|decrease).+increase[s.]*$)'
                    # the xxx increase, as xxx increases.
                    back_pattern = r'(?:increase.+as .+(?:increase|decrease))'
                    pattern = f'{straight_answer_pattern}|{front_pattern}|{back_pattern}'
                    if re.search(pattern, pred_text, re.I):
                        return True
                elif re.search(r'^decrease', gold_text, re.I):
                    straight_answer_pattern = r'(?:^decrease)'
                    # as xxx increases, the xxx increases.
                    front_pattern = r'(?:as .+(?:increase|decrease).+decrease[s.]*$)'
                    # the xxx increase, as xxx increases.
                    back_pattern = r'(?:decrease.+as .+(?:increase|decrease))'
                    pattern = f'{straight_answer_pattern}|{front_pattern}|{back_pattern}'
                    if re.search(pattern, pred_text, re.I):
                        return True
                elif re.search('^inverse', gold_text, re.I):
                    inverse_word_pattern = r'(?:\b(?:inversely|while|inverse)\b)'
                    inverse_sentence_pattern = r'(?:.+decrease.+as .+increase)|(?:.+increase.+as .+decrease)'
                    pattern = f'{inverse_word_pattern}|{inverse_sentence_pattern}'
                    if re.search(pattern, pred_text, re.I):
                        return True
                elif re.search(r'change as .+higher', raw_question, re.I):
                    if re.search('^higher', gold_text, re.I):
                        front_pattern = r'(?:(?:higher|sharper|increase).+as .+(?:higher|increase))'
                        back_pattern = r'(?:as .+(?:higher|increase).+(?:higher|sharper|increase))'
                        pattern = f'{front_pattern}|{back_pattern}'
                        if re.search(pattern, pred_text, re.I):
                            return True
        return False

    @staticmethod
    def string_in(pred_text, gold_text, raw_question, sub_task_name):
        # in most cases, we check gold if is included in prediction, as prediction tends to be longer
        if re.search(fr'(?:^| |\b){re.escape(gold_text)}\.?(?:\b| |$)', pred_text, re.I):
            # different from kie's narrative string,
            # we just check if gold is in prediction without length limit as prediction can be very long sometimes

            # fix false inclusion
            if sub_task_name == 'DUDE' and re.search(fr'\b{re.escape(gold_text)} or ', raw_question, re.I):
                if re.search(r'\bboth\b', pred_text, re.I):
                    return False
            return True

        # in some cases, we check if prediction in included in gold, as predictions from lazy models tends to be shorter
        selection_question = re.search(r'\b(?:which|where)\b', raw_question, re.I)
        shorter_prediction = len(pred_text) < len(gold_text)
        not_too_long_gold = gold_text.count(' ') <= 3
        if selection_question and shorter_prediction and not_too_long_gold:
            if re.search(fr'(?:^| |\b){re.escape(pred_text)}\.?(?:\b| |$)', gold_text, re.I):
                return True

        return False

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
                if match.group(1) == '.':
                    # remove the '.' for correct number parsing later
                    if re.search(r'[a-z]$', s[:match.start()], re.I) and re.search(r'^ ?\d', s[match.end() :], re.I):
                        detections.append((match.group(1), match.end(1) - 1, match.end()))
                else:
                    detections.append((match.group(1), match.end(1), match.end()))
        for r in detections:
            logger.debug(f'[Repetition] remove {r[0]} from string')
            s = s[:r[1]] + s[r[2]:]

        return s

    @staticmethod
    def preprocess_for_amounts_in_special_format(text):
        text_parts = text.split(',')
        text_parts[0] = text_parts[0].replace('.', ',')
        text = '.'.join(text_parts)
        return text

    @classmethod
    def rule_parse_numeric_value(
        cls, pred_text, gold_text, raw_question='', task_name='', sub_task_name='', is_evidence=False
    ):
        """
        Extract and parse numbers from string text. Carefully handle the following cases:
            + amounts where '.' and ',' are used reversely
            + missing or redundant percentage errors
            + extra 0 error
            + calculation error
            + sign error
            + unit error
            + etc
        """
        # cases checking
        possible_negative = 'decrease' in pred_text
        possible_thousands = ('$' in pred_text or re.search(',\d00|[1-9]000$', pred_text) is not None) and re.search(
            r'\bstars\b', raw_question, re.I
        ) is None

        interested_pred_text = cls.extract_number_part(pred_text, gold_text, raw_question, is_evidence)

        # handles amounts where '.' and ',' are used reversely for CORD dataset
        if sub_task_name == 'CORD':
            if re.search(r'^\D*\d{1,3}(?:\.\d{3})*(?:,\d{2})?\D*$', interested_pred_text):
                interested_pred_text = cls.preprocess_for_amounts_in_special_format(interested_pred_text)
            if re.search(r'^\D*\d{1,3}(?:\.\d{3})*(?:,\d{2})?\D*$', gold_text):
                gold_text = cls.preprocess_for_amounts_in_special_format(gold_text)

        pred_num = cls.parse_number(interested_pred_text)
        gold_num = cls.parse_number(gold_text)

        # handles percentage errors
        question_for_percentage = re.search(r'[a-z ](?:%|\(%\))[a-z ]|\b(?:percentage|percent)\b', raw_question, re.I)
        percent_sign_diff = (
            gold_text == interested_pred_text.replace('%', '')
            or f'{gold_text}%' in interested_pred_text
            or interested_pred_text == gold_text.replace('%', '')
            or f'{interested_pred_text}%' in gold_text
        )
        check_percentage = question_for_percentage or percent_sign_diff
        if pred_num.from_percentage and not gold_num.from_percentage and check_percentage:
            pred_num.restore_to_value_before_percentage()
        elif gold_num.from_percentage and not pred_num.from_percentage and check_percentage:
            if gold_num.maybe_percentage_error(pred_num):
                gold_num.restore_to_value_before_percentage()

        pred_text = pred_num.parsed_value
        gold_text = gold_num.parsed_value

        if '.' not in gold_text and re.search(r'\.0+$', pred_text):
            # extra 00 error
            pred_text = pred_num.get_int_str()
        elif pred_num.status and pred_num.decimal_num != gold_num.decimal_num:
            # extra decimal error and calculation error
            if task_name == 'Key Information Extraction':
                # decimal error in extraction is not acceptable
                round_num = max(pred_num.decimal_num, gold_num.decimal_num)
            else:
                # slight decimal error in calculation is acceptable
                min_decimal_num = min(pred_num.decimal_num, gold_num.decimal_num)
                not_all_percentage = pred_num.from_percentage != gold_num.from_percentage
                slight_error = gold_num.maybe_rounding_error(pred_num)
                if slight_error or not_all_percentage:
                    round_num = min_decimal_num
                else:
                    round_num = max(pred_num.decimal_num, gold_num.decimal_num)

            pred_text = pred_num.get_rounded_str(round_num)
            gold_text = gold_num.get_rounded_str(round_num)

        # ignore sign error in decrease setting
        if gold_text.startswith('-') and gold_text[1:] == pred_text and possible_negative:
            pred_text = f'-{pred_text}'
        # ignore sign error in subtraction questions
        elif (
            not is_evidence
            and 'difference' in raw_question
            and 'absolute' not in raw_question
            and gold_num.maybe_sign_error(pred_num)
        ):
            pred_text = gold_text

        # ignore thousands error
        elif possible_thousands and gold_num.maybe_thousands_error(pred_num):
            pred_text = gold_text
        return pred_text, gold_text

    @classmethod
    def rule_parse_extraction_value(
        cls, pred_text, gold_text, raw_question='', task_name='', sub_task_name='', is_evidence=False
    ):
        time_question = re.search('"(?:time_value)"', raw_question, re.I)
        phone_question = re.search('tel_value', raw_question, re.I)
        if gold_text.lower() in pred_text.lower() and len(pred_text) <= 2 * len(gold_text):
            # e.g., item name with irrelevant quantity
            # add restrictions to avoid false positive
            pred_text = gold_text
        elif re.sub(r"[ ']", '', gold_text).lower() == re.sub(r"[ ']", '', pred_text).lower():
            # extra ' and spaces
            pred_text = gold_text
        elif phone_question and re.sub(r'\D', '', pred_text) == re.sub(r'\D', '', gold_text):
            # 614-766-4494 is the same as (614) 766-4494
            pred_text = gold_text
        elif time_question:
            # remove noisy word and do not care the length restrictions
            pattern = r'time[: ]*'
            cleaned_gold_text = re.sub(
                r':(?=pm)', ' ', re.sub(pattern, '', gold_text, flags=re.I), flags=re.I
            ).replace(' ', '').lower()
            cleaned_pred_text = re.sub(
                r':(?=pm)', ' ', re.sub(pattern, '', pred_text, flags=re.I), flags=re.I
            ).replace(' ', '').lower()
            if cleaned_gold_text in cleaned_pred_text:
                pred_text = gold_text
        elif cls.is_same_date(pred_text, gold_text, raw_question):
            pred_text = gold_text

        return pred_text, gold_text

    @classmethod
    def rule_parse_narrative_value(
        cls, pred_text, gold_text, raw_question='', task_name='', sub_task_name='', is_evidence=False
    ):
        pred_text = normalize_narrative_text(pred_text, raw_question, sub_task_name)
        gold_text = normalize_narrative_text(gold_text, raw_question, sub_task_name)
        if cls.is_same_ordinal_number(pred_text, gold_text, raw_question):
            pred_text = gold_text
        elif cls.is_same_comparative_adjective(pred_text, gold_text, raw_question):
            pred_text = gold_text
        elif cls.is_same_trend(pred_text, gold_text, raw_question, task_name, sub_task_name):
            pred_text = gold_text
        elif cls.string_in(pred_text, gold_text, raw_question, sub_task_name):
            pred_text = gold_text
        elif gold_text.replace(' ', '').lower() == pred_text.replace(' ', '').lower():
            # space errors
            pred_text = gold_text
        elif gold_text.count(' ') >= 3 or (' ' in gold_text and len([i for i in gold_text if i.isalpha()]) >= 15):
            # long answer with disturbing context information
            # tolerant on spelling errors

            # the original prediction
            candidates = [pred_text]

            # add another candidate((seperated by dot or 'including', 'is' keywords))
            seps = ['. ', ', including', 'is']
            for seperator in seps:
                prediction_parts = pred_text.split(seperator)
                if len(prediction_parts) > 1:
                    candidates.append(prediction_parts[0])

            # check if any
            for pred_candidate in candidates:
                if anls_compute(gold_text, pred_candidate) < 0.5:
                    pred_text = gold_text
                    break

        return pred_text, gold_text

    @classmethod
    def rule_parse(cls, pred_text, gold_text, raw_question='', task_name='', sub_task_name='', is_evidence=False):
        """
        Check if pred_text expresses the same meaning as gold_text. If so, update pred_text to gold_text.

        1. number comparison
            1.1 float/int only
            1.2 percentage
            1.3 number + units
            1.4 currency + number
        2. yes or no
        3. long answer with disturbing context information
        """
        if cls.is_gold_number(gold_text):
            pred_text, gold_text = cls.rule_parse_numeric_value(
                pred_text=pred_text,
                gold_text=gold_text,
                raw_question=raw_question,
                task_name=task_name,
                sub_task_name=sub_task_name,
                is_evidence=is_evidence
            )
        elif cls.is_same_judgement(pred_text, gold_text):
            pred_text = gold_text
        elif is_narrative_kie(task_name, raw_question):
            pred_text, gold_text = cls.rule_parse_extraction_value(
                pred_text=pred_text,
                gold_text=gold_text,
                raw_question=raw_question,
                task_name=task_name,
                sub_task_name=sub_task_name,
                is_evidence=is_evidence
            )
        elif 'Question Answering' in task_name or (is_reasoning_task(task_name)):
            pred_text, gold_text = cls.rule_parse_narrative_value(
                pred_text=pred_text,
                gold_text=gold_text,
                raw_question=raw_question,
                task_name=task_name,
                sub_task_name=sub_task_name,
                is_evidence=is_evidence
            )

        return pred_text, gold_text

    @classmethod
    def parse(cls, pred_text, gold_text, raw_question='', task_name='', sub_task_name='', is_evidence=False):
        pred_text = cls.to_string(pred_text)
        gold_text = cls.to_string(gold_text)

        pred_text = cls.remove_repetitions(pred_text)

        skip_parse = re.search(r'Recognition|Localization|Forgery', task_name) is not None
        keep_newline = 'Recognition' in task_name
        skip_normalize = False
        if not skip_parse and pred_text is not None and gold_text is not None:
            if cls.is_gold_number(gold_text):
                skip_normalize = True

            if pred_text != gold_text:
                pred_text, gold_text = cls.rule_parse(
                    pred_text=pred_text,
                    gold_text=gold_text,
                    raw_question=raw_question,
                    task_name=task_name,
                    sub_task_name=sub_task_name,
                    is_evidence=is_evidence
                )
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
