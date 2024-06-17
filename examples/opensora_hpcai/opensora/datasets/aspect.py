"""
Credit: OpenSora HPC-AI Tech
https://github.com/hpcaitech/Open-Sora/blob/ea41df3d6cc5f389b6824572854d97fa9f7779c3/opensora/datasets/aspect.py
"""

# S = 8294400
ASPECT_RATIO_4K = {
    "0.38": (1764, 4704),
    "0.43": (1886, 4400),
    "0.48": (1996, 4158),
    "0.50": (2036, 4072),
    "0.53": (2096, 3960),
    "0.54": (2118, 3918),
    "0.62": (2276, 3642),
    "0.56": (2160, 3840),  # base
    "0.67": (2352, 3528),
    "0.75": (2494, 3326),
    "1.00": (2880, 2880),
    "1.33": (3326, 2494),
    "1.50": (3528, 2352),
    "1.78": (3840, 2160),
    "1.89": (3958, 2096),
    "2.00": (4072, 2036),
    "2.08": (4156, 1994),
}

# S = 2073600
ASPECT_RATIO_1080P = {
    "0.38": (882, 2352),
    "0.43": (942, 2198),
    "0.48": (998, 2080),
    "0.50": (1018, 2036),
    "0.53": (1048, 1980),
    "0.54": (1058, 1958),
    "0.56": (1080, 1920),  # base
    "0.62": (1138, 1820),
    "0.67": (1176, 1764),
    "0.75": (1248, 1664),
    "1.00": (1440, 1440),
    "1.33": (1662, 1246),
    "1.50": (1764, 1176),
    "1.78": (1920, 1080),
    "1.89": (1980, 1048),
    "2.00": (2036, 1018),
    "2.08": (2078, 998),
}

# S = 921600
ASPECT_RATIO_720P = {
    "0.38": (588, 1568),
    "0.43": (628, 1466),
    "0.48": (666, 1388),
    "0.50": (678, 1356),
    "0.53": (698, 1318),
    "0.54": (706, 1306),
    "0.56": (720, 1280),  # base
    "0.62": (758, 1212),
    "0.67": (784, 1176),
    "0.75": (832, 1110),
    "1.00": (960, 960),
    "1.33": (1108, 832),
    "1.50": (1176, 784),
    "1.78": (1280, 720),
    "1.89": (1320, 698),
    "2.00": (1358, 680),
    "2.08": (1386, 666),
}

# S = 409920
ASPECT_RATIO_480P = {
    "0.38": (392, 1046),
    "0.43": (420, 980),
    "0.48": (444, 925),
    "0.50": (452, 904),
    "0.53": (466, 880),
    "0.54": (470, 870),
    "0.56": (480, 854),  # base
    "0.62": (506, 810),
    "0.67": (522, 784),
    "0.75": (554, 738),
    "1.00": (640, 640),
    "1.33": (740, 555),
    "1.50": (784, 522),
    "1.78": (854, 480),
    "1.89": (880, 466),
    "2.00": (906, 454),
    "2.08": (924, 444),
}

# S = 230400
ASPECT_RATIO_360P = {
    "0.38": (294, 784),
    "0.43": (314, 732),
    "0.48": (332, 692),
    "0.50": (340, 680),
    "0.53": (350, 662),
    "0.54": (352, 652),
    "0.56": (360, 640),  # base
    "0.62": (380, 608),
    "0.67": (392, 588),
    "0.75": (416, 554),
    "1.00": (480, 480),
    "1.33": (554, 416),
    "1.50": (588, 392),
    "1.78": (640, 360),
    "1.89": (660, 350),
    "2.00": (678, 340),
    "2.08": (692, 332),
}

# S = 102240
ASPECT_RATIO_240P = {
    "0.38": (196, 522),
    "0.43": (210, 490),
    "0.48": (222, 462),
    "0.50": (226, 452),
    "0.53": (232, 438),
    "0.54": (236, 436),
    "0.56": (240, 426),  # base
    "0.62": (252, 404),
    "0.67": (262, 393),
    "0.75": (276, 368),
    "1.00": (320, 320),
    "1.33": (370, 278),
    "1.50": (392, 262),
    "1.78": (426, 240),
    "1.89": (440, 232),
    "2.00": (452, 226),
    "2.08": (462, 222),
}

# S = 36864
ASPECT_RATIO_144P = {
    "0.38": (117, 312),
    "0.43": (125, 291),
    "0.48": (133, 277),
    "0.50": (135, 270),
    "0.53": (139, 262),
    "0.54": (141, 260),
    "0.56": (144, 256),  # base
    "0.62": (151, 241),
    "0.67": (156, 234),
    "0.75": (166, 221),
    "1.00": (192, 192),
    "1.33": (221, 165),
    "1.50": (235, 156),
    "1.78": (256, 144),
    "1.89": (263, 139),
    "2.00": (271, 135),
    "2.08": (277, 132),
}

# from PixArt
# S = 1048576
ASPECT_RATIO_1024 = {
    "0.25": (512, 2048),
    "0.26": (512, 1984),
    "0.27": (512, 1920),
    "0.28": (512, 1856),
    "0.32": (576, 1792),
    "0.33": (576, 1728),
    "0.35": (576, 1664),
    "0.4": (640, 1600),
    "0.42": (640, 1536),
    "0.48": (704, 1472),
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
    "3.11": (1792, 576),
    "3.62": (1856, 512),
    "3.75": (1920, 512),
    "3.88": (1984, 512),
    "4.0": (2048, 512),
}

# S = 262144
ASPECT_RATIO_512 = {
    "0.25": (256, 1024),
    "0.26": (256, 992),
    "0.27": (256, 960),
    "0.28": (256, 928),
    "0.32": (288, 896),
    "0.33": (288, 864),
    "0.35": (288, 832),
    "0.4": (320, 800),
    "0.42": (320, 768),
    "0.48": (352, 736),
    "0.5": (352, 704),
    "0.52": (352, 672),
    "0.57": (384, 672),
    "0.6": (384, 640),
    "0.68": (416, 608),
    "0.72": (416, 576),
    "0.78": (448, 576),
    "0.82": (448, 544),
    "0.88": (480, 544),
    "0.94": (480, 512),
    "1.0": (512, 512),
    "1.07": (512, 480),
    "1.13": (544, 480),
    "1.21": (544, 448),
    "1.29": (576, 448),
    "1.38": (576, 416),
    "1.46": (608, 416),
    "1.67": (640, 384),
    "1.75": (672, 384),
    "2.0": (704, 352),
    "2.09": (736, 352),
    "2.4": (768, 320),
    "2.5": (800, 320),
    "2.89": (832, 288),
    "3.0": (864, 288),
    "3.11": (896, 288),
    "3.62": (928, 256),
    "3.75": (960, 256),
    "3.88": (992, 256),
    "4.0": (1024, 256),
}

# S = 65536
ASPECT_RATIO_256 = {
    "0.25": (128, 512),
    "0.26": (128, 496),
    "0.27": (128, 480),
    "0.28": (128, 464),
    "0.32": (144, 448),
    "0.33": (144, 432),
    "0.35": (144, 416),
    "0.4": (160, 400),
    "0.42": (160, 384),
    "0.48": (176, 368),
    "0.5": (176, 352),
    "0.52": (176, 336),
    "0.57": (192, 336),
    "0.6": (192, 320),
    "0.68": (208, 304),
    "0.72": (208, 288),
    "0.78": (224, 288),
    "0.82": (224, 272),
    "0.88": (240, 272),
    "0.94": (240, 256),
    "1.0": (256, 256),
    "1.07": (256, 240),
    "1.13": (272, 240),
    "1.21": (272, 224),
    "1.29": (288, 224),
    "1.38": (288, 208),
    "1.46": (304, 208),
    "1.67": (320, 192),
    "1.75": (336, 192),
    "2.0": (352, 176),
    "2.09": (368, 176),
    "2.4": (384, 160),
    "2.5": (400, 160),
    "2.89": (416, 144),
    "3.0": (432, 144),
    "3.11": (448, 144),
    "3.62": (464, 128),
    "3.75": (480, 128),
    "3.88": (496, 128),
    "4.0": (512, 128),
}


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return closest_ratio


ASPECT_RATIOS = {
    "144p": (36864, ASPECT_RATIO_144P),
    "256": (65536, ASPECT_RATIO_256),
    "240p": (102240, ASPECT_RATIO_240P),
    "360p": (230400, ASPECT_RATIO_360P),
    "512": (262144, ASPECT_RATIO_512),
    "480p": (409920, ASPECT_RATIO_480P),
    "720p": (921600, ASPECT_RATIO_720P),
    "1024": (1048576, ASPECT_RATIO_1024),
    "1080p": (2073600, ASPECT_RATIO_1080P),
    "4k": (8294400, ASPECT_RATIO_4K),
}
