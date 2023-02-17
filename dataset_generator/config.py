from fvcore.common.config import CfgNode

_C = CfgNode()

# Data settings
_C.DATA = CfgNode()
_C.DATA.IMG_SIZE = 361

# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS = [7, 9]
_C.PREPROCESS.FACE_CLASS = [1, 6]
_C.PREPROCESS.EYEBROW_CLASS = [2, 3]
_C.PREPROCESS.EYE_CLASS = [4, 5]
_C.PREPROCESS.EAR_CLASS = [11, 12]
_C.PREPROCESS.NECK_CLASS = [13]

# Pseudo ground truth
_C.PGT = CfgNode()
_C.PGT.EYE_MARGIN = 24
_C.PGT.LIP_MARGIN = 4
_C.PGT.SKIN_ALPHA = 0.5
_C.PGT.EYE_ALPHA = 0.9
_C.PGT.LIP_ALPHA = 0.6

def get_config()->CfgNode:
    return _C
