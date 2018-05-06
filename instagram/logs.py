import logging

# from data_repo import d0
from user_config_reader import load_secrets

# d = d0
# u = d.u
u = load_secrets()[0]

u_format = '(%s): ' % (u)
formatter = logging.Formatter(
    u_format +
    '[%(asctime)s] - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

fh = logging.FileHandler('/data/ig/logs/instagram-%s.log' % u)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)


fh2 = logging.FileHandler('/data/ig/logs/instagram.log')
fh2.setLevel(logging.DEBUG)
fh2.setFormatter(formatter)

logger = logging.getLogger('InstagramBot')
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)
logger.addHandler(fh2)


def main():
    logger.info('Test: This is an INFO')
    logger.error("Test: This is ERROR")
    try:
        raise Exception("error message")
    except Exception as e:
        logger.error("Test: Type %s", type(e))
        logger.error("Test: Args %s", e.args)
        logger.error(e)


if __name__ == '__main__':
    main()
