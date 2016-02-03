'''
Created on Jan 27, 2016

@author: Roman Feldbauer
'''

import sys, time

class ConsoleLogging():
    
    @property
    def _current_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')
    
    def message(self, *objs, flush=False):
        print(self._current_time, 'INFO:', *objs)
        if flush:
            sys.stdout.flush()
        
    def warning(self, *objs, flush=False):
        print(self._current_time, 'WARNING:', *objs, file=sys.stderr)
        if flush:
            sys.stderr.flush()
        
    def error(self, *objs, flush=False):
        print(self._current_time, 'ERROR:', *objs, file=sys.stderr)
        if flush:
            sys.stderr.flush()
        
if __name__ == '__main__':
    log = ConsoleLogging()
    log.message('This module supplies functions for printing and logging.')
    log.message('Examples:')
    sys.stdout.flush()
    time.sleep(0.01)
    log.warning('This is a warning.')
    log.error('This is an error!')
    sys.stderr.flush()
    time.sleep(0.01)
    log.message('You should have got three messages on stdout and two on stderr.')