#ログ制御用クラス
import datetime
import logging
import optuna
import os


class ControlLog(object):
    def __init__(self, log_file_name=None, log_dir_path=None):
        #loggerのhandlerを削除しないといけないのでインスタンス化している
        self.logger = None
        self.log_fhandler = None
        self.log_file_name = log_file_name
        self.log_dir_path = log_dir_path
        
    def decide_filename(self, what_log="OptunaLogs"):
        dt_today = datetime.date.today()
        dt_now = datetime.datetime.now()
        flog_dir_name = str(dt_today) + "/"
        flog_file_name = str(dt_now.hour) + str(dt_now.minute) + str(dt_now.second)

        #ログ出力を行うディレクトリ作成
        flog_dir_path = "../" + what_log + "/" + flog_dir_name
        os.makedirs(flog_dir_path, exist_ok=True)

        return flog_file_name, flog_dir_path
    
    def set_log_filehandler(self):
        #ロガー等の作成
        #新しいログを出力するための参照先設定
        if not self.log_file_name:
            self.log_file_name, self.log_dir_path = self.decide_filename()
        
        # print(logger_name, type(logger_name))
        self.logger = logging.getLogger("optuna")
        self.logger.setLevel(logging.INFO)

        self.log_fhandler = logging.FileHandler(self.log_dir_path+self.log_file_name+".txt", mode="a")
        self.log_fhandler.setLevel(logging.INFO)

        self.logger.addHandler(self.log_fhandler)

        #コンソールへのログの表示を無効化する(デフォルトでoptunaに指定されているハンドラを無効化する．しないとコンソールに出力され続ける)
        optuna.logging.disable_default_handler()
        #ログの表示内容を設定する
        # optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        print(f"現在のログレベル: {optuna.logging.get_verbosity()}")
        
        #optunaによる探索結果を格納するDBファイルのURIを指定
        sqlite_path = "sqlite:///" + self.log_dir_path + "optuna.sqlite3"
        print(f"SQLite file path: {sqlite_path}")
        
        return sqlite_path

    #handlerの削除をしないと，前のログファイルにもアクセスしてしまう．
    #loggerの削除をしてしまうと，同じloggerではログが上手く取得できなくなってしまう

    def kill_logger(self):
        """
        loggerを削除する
        logger：削除したいロガー(logging.Logger)
        """
        name = self.logger.name
        del logging.Logger.manager.loggerDict[name]

        return

    def kill_handler(self):
        """
        loggerから特定のハンドルを削除する
        logger：ハンドルを削除したいロガー(logging.Logger)
        handles：削除したいハンドル(list)
        """
        
        handles = [self.log_fhandler]
        for handle in handles:
            self.logger.removeHandler(handle)

        return
