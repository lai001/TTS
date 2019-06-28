'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in Symbols.py to match your data).
'''

import re
from unidecode import unidecode
from TextProcess.Numbers import Numbers

class Cleaner:
    def __init__(self):
        # Regular expression matching whitespace:
        self._whitespace_re = re.compile(r'\s+')

        # List of (regular expression, replacement) pairs for abbreviations:
        self._abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
            ('mrs', 'misess'),
            ('mr', 'mister'),
            ('dr', 'doctor'),
            ('st', 'saint'),
            ('co', 'company'),
            ('jr', 'junior'),
            ('maj', 'major'),
            ('gen', 'general'),
            ('drs', 'doctors'),
            ('rev', 'reverend'),
            ('lt', 'lieutenant'),
            ('hon', 'honorable'),
            ('sgt', 'sergeant'),
            ('capt', 'captain'),
            ('esq', 'esquire'),
            ('ltd', 'limited'),
            ('col', 'colonel'),
            ('ft', 'fort'),
        ]]

        self.cleaners = {"basic_cleaners": self.basic_cleaners,
                         "english_cleaners": self.english_cleaners,
                         "transliteration_cleaners": self.transliteration_cleaners
                         }

    def expand_abbreviations(self, text: str):
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def expand_numbers(self, text: str):

        return Numbers().normalize_numbers(text)

    def lowercase(self, text: str):
        return text.lower()

    def collapse_whitespace(self, text: str):
        return re.sub(self._whitespace_re, ' ', text)

    def convert_to_ascii(self, text: str):
        return unidecode(text)

    def basic_cleaners(self, text: str):
        """Basic pipeline that lowercases and collapses whitespace without transliteration."""
        text = self.lowercase(text)
        text = self.collapse_whitespace(text)
        return text

    def transliteration_cleaners(self, text: str):
        """Pipeline for non-English text that transliterates to ASCII."""
        text = self.convert_to_ascii(text)
        text = self.lowercase(text)
        text = self.collapse_whitespace(text)
        return text

    def english_cleaners(self, text: str):
        """Pipeline for English text, including number and abbreviation expansion."""
        text = self.convert_to_ascii(text)
        text = self.lowercase(text)
        text = self.expand_numbers(text)
        text = self.expand_abbreviations(text)
        text = self.collapse_whitespace(text)
        return text

    def clean_text(self, text: str, cleaner_names: list):
        for name in cleaner_names:
            cleaner = self.cleaners[name]
            if not cleaner:
                raise Exception('Unknown cleaner: %s' % name)
            text = cleaner(text)
        return text
