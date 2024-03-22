# -*- coding: utf8 -*-
from collections import namedtuple

DATA_PATH_12 = '/home/lipu/smed/datasets/Twitter'
COLUMNS_12 = ["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", "place_type",
              "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls",
              "entities", "words", "filtered_words", "sampled_words"]
DataItem12 = namedtuple('DataItem12', COLUMNS_12)

DATA_PATH_18 = '/home/lipu/smed/datasets/Twitter_2018'
COLUMNS_18 = ["tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
              "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
              "sampled_words"]
DataItem18 = namedtuple('DataItem18', COLUMNS_18)




SAMPLE_NUM_TWEET = 60

WINDOW_SIZE = 3

POS_word_tokens = [
    31642,
    40496,
    38610,
    28039,
    42096,
    10537,
    31428,
    2092,
    11014,
    12470,
    15298,
    18925,
    20594,
    26789,
    35712,
    8187,
    40582,
    5884,
    15236,
    19809,
    20112,
    3519,
    5363,
    9819,
    22706,
    42225,
    6692,
    25614,
    29014,
    32852,
    3917,
    10575,
    29474,
    48135,
    12336,
    40255,
    7548,
    22849,
    43457,
]

NEG_word_tokens = [
    7310,
    12941,
    18778,
    30911,
    36640,
    39799,
    40341,
    1180,
    10338,
    20615,
    22577,
    28754,
    19938,
    37433,
    10084,
    11266,
    13869,
    4553,
    40737,
    28810,
    6697,
    29096,
    27294,
    15641,
    48621,
    39235,
    18778,
    34750,
    40566,
    4795,
    13362,
    14799,
    18326,
    43607,
    10388,
    5023,
    7485,
    12101,
    18521,
    22596,
    39182,
    5475,
    11266,
    37334,
]
PMT_word_token = [ 8491, 262, 734, 13439, 2029, 2092, 25]
