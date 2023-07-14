"""
Define log manager
TODO Write docs
"""
import datetime
import logging
import os

import optuna


class LogModel(object):
    """
    Base log model
    """

    def __init__(self, log_dir_path=None, log_file_name=None):
        self.logger = None
        self.handler = None
        if not log_dir_path:
            log_dir_path, _ = self.decide_filename()
        if not log_file_name:
            _, log_file_name = self.decide_filename()
        self.log_dir_path = log_dir_path
        self.log_file_name = log_file_name
        
    @staticmethod
    def decide_filename(what_log="."):
        """
        Naming and making the file from which the log will be output
        This code gets Date and time from python datetime module.

        Parameters
        ----------
        what_log : str, default="."
            Set directory name where log will be output

        Returns
        -------
        log_file_name : str
            File name is named after the current time.

        log_dir_path : str
            Log directory name = "user define name" / "today's Date"
        """

        dt_today = str(datetime.date.today()) + "/"
        dt_now = datetime.datetime.now()
        log_dir_path = what_log + "/" + dt_today
        log_file_name = str(dt_now.hour) + str(dt_now.minute) + str(dt_now.second)
        os.makedirs(log_dir_path, exist_ok=True)
        return log_dir_path, log_file_name

    def kill_logger(self):
        name = self.logger.name
        del logging.Logger.manager.loggerDict[name]

    def kill_handler(self):
        handles = [self.handler]
        for handle in handles:
            self.logger.removeHandler(handle)
    

class OptunaLogModel(LogModel):
    """Definition of class optuna log model"""

    def __init__(self, log_dir_path=None, log_file_name=None):
        super().__init__(log_dir_path=log_dir_path, log_file_name=log_file_name)
            
    def set_sqlite_path(self):
        # Must set logger name "optuna" to get optuna log
        self.logger = logging.getLogger("optuna")
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(
            self.log_dir_path+self.log_file_name+".txt", mode="a"
        )
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        # Disable the display of logs on the console.
        optuna.logging.disable_default_handler()
        # Set log level of optuna
        # optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        # print(f"Current log level: {optuna.logging.get_verbosity()}")
        sqlite_path = "sqlite:///" + self.log_dir_path + "optuna.sqlite3"
        print(f"SQLite file path: {sqlite_path}")
        return sqlite_path
