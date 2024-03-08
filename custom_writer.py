import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry, BasicWriter


class MyCustomWriter(BasicWriter):
    def __init__(
            self,
            *args, **kwargs
    ):
        print(f"args {args}")
        print(f"kwargs {kwargs}")

        super(MyCustomWriter, self).__init__(*args, **kwargs)

        print("output dir is ", self._output_dir)

        self.frame_cnt = 0

    def write(self, data: dict):
        # 降低帧率
        self.frame_cnt += 1
        if self.frame_cnt % 5 != 0:
            return
        super().write(data)


WriterRegistry.register(MyCustomWriter)
