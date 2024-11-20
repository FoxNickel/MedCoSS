from log_util import LogUtil

logger = LogUtil.get_logger("demo")

logger.debug('这是一个调试信息')
logger.info('这是一个信息')
logger.warning('这是一个警告')
logger.error('这是一个错误')
logger.critical('这是一个严重错误')