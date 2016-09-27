#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2015-2016, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import sys, time
from abc import ABCMeta, abstractmethod

class Logging(metaclass=ABCMeta): # pragma: no cover
    """Base class for time-stamped logging.
    
    Do not instantiate this class, but ConsoleLogging or FileLogging!
    """
    @property
    def _current_time(self):
        """Formatted time stamp"""
        return time.strftime('%Y-%m-%d %H:%M:%S')
    
    @abstractmethod
    def message(self):
        ...
    @abstractmethod
    def warning(self):
        ...
    @abstractmethod
    def error(self):
        ...
    
class ConsoleLogging(Logging):
    """Convenience functions for time-stamped logging to the console"""
    
    def message(self, *objs, flush=True):
        """Log normal program function"""
        print(self._current_time, 'INFO:', *objs)
        if flush:
            sys.stdout.flush()
        
    def warning(self, *objs, flush=True):
        """Log warning (program can still continue)"""
        print(self._current_time, 'WARNING:', *objs, file=sys.stderr)
        if flush:
            sys.stderr.flush()
        
    def error(self, *objs, flush=True):
        """Log error (program fails)"""
        print(self._current_time, 'ERROR:', *objs, file=sys.stderr)
        if flush:
            sys.stderr.flush()
            
class FileLogging(ConsoleLogging):
    """Convenience functions for time-stamped logging to a file"""
    
    def __init__(self):
        """Not implemented"""
        self.warning("FileLogging not yet implemented, will print to "
                     "console anyway.")
        
if __name__ == '__main__':
    """Simple test of this module"""
    log = ConsoleLogging()
    log.message('This module supplies functions for printing and logging.')
    log.message('Examples:')
    sys.stdout.flush()
    time.sleep(0.01)
    log.warning('This is a warning.')
    log.error('This is an error!')
    sys.stderr.flush()
    time.sleep(0.01)
    log.message('You should have got three messages on stdout and '
                'two on stderr.')
    log = FileLogging()
    log.message('Still written to console, until implemented.')
    try:
        log = Logging()
    except TypeError as e:
        log.warning('Must not instantiate Logging(), got exception:\n', e)
        