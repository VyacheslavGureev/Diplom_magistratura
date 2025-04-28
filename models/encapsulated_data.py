# Данные про датасет в одном месте. Решение в стиле простенького ООП
class EncapsulatedDataloaders:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test


class EncapsulatedDataloadersTextDescr(EncapsulatedDataloaders):
    def __init__(self, train, val, test, text_descr):
        super().__init__(train, val, test)
        self.text_descr = text_descr