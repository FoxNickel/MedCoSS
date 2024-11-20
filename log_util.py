import logging
import logging.config
import inspect

class LogUtil:
    @staticmethod
    def setup_logging(log_file='app.log'):
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                },
            },
            'handlers': {
                'console': {
                    'level': 'DEBUG',
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                },
                'file': {
                    'level': 'DEBUG',
                    'class': 'logging.FileHandler',
                    'filename': log_file,
                    'formatter': 'standard',
                },
            },
            'root': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
            },
        })

    @staticmethod
    def get_logger(module_name):
        return logging.getLogger(module_name)

# 初始化日志配置
LogUtil.setup_logging()