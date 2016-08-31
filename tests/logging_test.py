#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2016, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import unittest
from hub_toolbox.Logging import Logging, ConsoleLogging, FileLogging

class TestLogging(unittest.TestCase):
    """Minimally test Logging (should switch to std module logging anyway)"""

    def test_unable_to_instantiate_abstract_class_logging(self):
        with self.assertRaises(TypeError):
            Logging()

    def test_console_logging_has_all_methods(self):
        log = ConsoleLogging()
        has_all_attributes = hasattr(log, 'warning') and \
            hasattr(log, 'warning') and hasattr(log, 'error')
        return self.assertTrue(has_all_attributes)

    def test_file_logging_has_all_methods(self):
        log = FileLogging()
        has_all_attributes = hasattr(log, 'warning') and \
            hasattr(log, 'warning') and hasattr(log, 'error')
        return self.assertTrue(has_all_attributes)

    def test_message(self):
        log = ConsoleLogging()
        log.message("Message")
        return self

    def test_warning(self):
        log = ConsoleLogging()
        log.warning("Warning")
        return self

    def test_error(self):
        log = ConsoleLogging()
        log.error("Error")
        return self

if __name__ == "__main__":
    unittest.main()
