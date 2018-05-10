import sys

class FilterConfig():

    def __init__(self):
        self.min_word_count = 0
        self.min_timespent_minutes = 0
        self.max_timespent_minutes = sys.maxsize
        self.min_project_size = 0
        self.even_distribution_bin_count = 0

    def set_param(self, param_name, param_value):
        setattr(self, param_name, param_value)