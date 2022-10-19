from ..builder import PIPELINES
import pandas as pd
import mmcv

__all__ = ["LoadSignalFromFile", "SignalHandler"]


@mmcv.register_handler('signal')
class SignalHandler(mmcv.BaseFileHandler):

    def load_from_fileobj(self, file, nrows):
        data = pd.read_csv(file, index_col=0, nrows=nrows)
        return data

    def dump_to_fileobj(self, data, file):
        data.to_csv(file)

    def dump_to_str(self, obj, **kwargs):
        return str(obj)


@PIPELINES.register_module()
class LoadSignalFromFile:
    """
    Load an signal from file.
    """

    def __init__(
        self,
        classes,
        upper=3001,
        skiprows=1,
        nrows=None,
    ):
        self.nrows = nrows
        self.skiprows = skiprows
        self.upper = upper
        self.classes = classes
        self._classname_to_classidx = dict(zip(self.classes, range(len(self.classes))))

    def __call__(self, results):
        data = mmcv.load(results['path'], nrows=self.nrows)
        results['dtoas'] = data['DTOA'].clip(lower=1, upper=self.upper).tolist()[self.skiprows:]
        if "MODE" in data.columns:
            results['modes'] = [self._classname_to_classidx[mode] for mode in data['MODE'].tolist()[self.skiprows:]]
        else:
            results['modes'] = None
        if "TAG" in data.columns:
            results['tags'] = data['TAG'].tolist()[self.skiprows:]
        else:
            results['tags'] = None
        return results

    def __repr__(self):
        return self.__class__.__name__