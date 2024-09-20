# -*- coding: utf-8 -*-

OHE_COLS=['p_arrive_poi1', 'p_arrive_poi2', 'p_arrive_poi3',
                 'p_arrive_poi1_weekday','p_arrive_poi2_weekday', 'p_arrive_poi3_weekday',
                 'p_arrive_poi1_weekend', 'p_arrive_poi2_weekend', 'p_arrive_poi3_weekend']

LBE_COLS = [ ]

DROP_COLS = [
    'lifecycle',
    'city_level',
    'touch_chanel',
    'dt',
    'his_resident_city_id',
    'last_bubble_city',
    'resident_city_id',
    'common_county_id',
    'county_name',
    'region',
    'before_finish_time',
    'treatment_group',
    'pid'
]


LABEL_COL = 'if_send'
TREATMENT_COL = 'treatment_group'
COST_COL = 'reward'
control_name = 'control'
