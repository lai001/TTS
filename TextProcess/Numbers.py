import inflect
import re


class Numbers:

    def __init__(self):
        self._inflect = inflect.engine()
        self._comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
        self._decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
        self._pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
        self._dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
        self._ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
        self._number_re = re.compile(r'[0-9]+')

    def _remove_commas(self, m):
        return m.group(1).replace(',', '')

    def _expand_decimal_point(self, m):
        return m.group(1).replace('.', ' point ')

    def _expand_dollars(self, m):
        match = m.group(1)
        parts = match.split('.')
        if len(parts) > 2:
            return match + ' dollars'  # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            cent_unit = 'cent' if cents == 1 else 'cents'
            return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            return '%s %s' % (dollars, dollar_unit)
        elif cents:
            cent_unit = 'cent' if cents == 1 else 'cents'
            return '%s %s' % (cents, cent_unit)
        else:
            return 'zero dollars'

    def _expand_ordinal(self, m):
        return self._inflect.number_to_words(m.group(0))

    def _expand_number(self, m):
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return 'two thousand'
            elif num > 2000 and num < 2010:
                return 'two thousand ' + self._inflect.number_to_words(num % 100)
            elif num % 100 == 0:
                return self._inflect.number_to_words(num // 100) + ' hundred'
            else:
                return self._inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
        else:
            return self._inflect.number_to_words(num, andword='')

    def normalize_numbers(self, text):
        text = re.sub(self._comma_number_re, self._remove_commas, text)
        text = re.sub(self._pounds_re, r'\1 pounds', text)
        text = re.sub(self._dollars_re, self._expand_dollars, text)
        text = re.sub(self._decimal_number_re, self._expand_decimal_point, text)
        text = re.sub(self._ordinal_re, self._expand_ordinal, text)
        text = re.sub(self._number_re, self._expand_number, text)
        return text
