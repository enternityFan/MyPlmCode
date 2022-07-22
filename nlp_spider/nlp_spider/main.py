# @Time : 2022-07-08 17:20
# @Author : Phalange
# @File : main.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

from scrapy.cmdline import execute

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
execute(["scrapy","crawl","sunxiaochuan"])#启动文件
