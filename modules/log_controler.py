#ログ制御用クラス
import datetime
import logging
import optuna
import os

class ControlLog:
    def __init__(self):
        #loggerのhandlerを削除しないといけないのでインスタンス化している
        self.logger = None
        self.log_fhandler = None
        
    def decide_filename(self):
        dt_today = datetime.date.today()
        dt_now = datetime.datetime.now()
        log_dir_name = str(dt_today) + "/"
        log_file_name = str(dt_now.hour) + str(dt_now.minute) + str(dt_now.second)

        #ログ出力を行うディレクトリ作成
        log_dir_path = "./OptunaLogs/" + log_dir_name
        os.makedirs(log_dir_path, exist_ok=True)
        
        return log_file_name, log_dir_path
    
    def set_log(self, log_file_name=0, log_dir_path=0):
        #ロガー等の作成
        #新しいログを出力するための参照先設定
        if log_file_name == 0:
            log_file_name, log_dir_path = self.decide_filename()
        
        self.logger = logging.getLogger("optuna")
        self.logger.setLevel(logging.INFO)

        self.log_fhandler = logging.FileHandler(log_dir_path+log_file_name+".txt", mode="a")
        self.log_fhandler.setLevel(logging.INFO)

        self.logger.addHandler(self.log_fhandler)

        #コンソールへのログの表示を無効化する(デフォルトでoptunaに指定されているハンドラを無効化する．しないとコンソールに出力され続ける)
        optuna.logging.disable_default_handler()
        #ログの表示内容を設定する
        # optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        print(f"現在のログレベル: {optuna.logging.get_verbosity()}")
        
        #optunaによる探索結果を格納するDBファイルのURIを指定
        sqlite_path = "sqlite:///" + log_dir_path + "optuna.sqlite3"
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
