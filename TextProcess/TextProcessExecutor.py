import re

from TextProcess.Cleaners import Cleaner
from TextProcess.Symbols import Symbols

class TextProcessExecutor:
    def __init__(self):
        self.curly_re = re.compile(r"(.*?){(.+?)}(.*)")

        self._id_to_symbol = {i: s for i, s in enumerate(Symbols.symbols)}
        self._symbol_to_id = {s: i for i, s in enumerate(Symbols.symbols)}
        self.cleaners = Cleaner()

    def _symbols_to_sequence(self, symbols):
        return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(['@' + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self._symbol_to_id and s is not '_' and s is not '~'

    def sequence_text(self, text, cleaner_names=None):
        if cleaner_names is None:
            cleaner_names = ["english_cleaners"]
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            match = self.curly_re.match(text)
            if not match:
                sequence += self._symbols_to_sequence(self.cleaners.clean_text(text, cleaner_names))
                break

            sequence += self._symbols_to_sequence(self.cleaners.clean_text(match.group(1), cleaner_names))
            sequence += self._arpabet_to_sequence(match.group(2))

            text = match.group(3)

        # Append EOS token
        sequence.append(self._symbol_to_id['~'])
        return sequence
