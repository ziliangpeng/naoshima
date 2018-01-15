import logging

import data_repo

d = data_repo.datas[0]
u = d.u

formatter = logging.Formatter(u + ': [%(asctime)s] - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

fh = logging.FileHandler('/data/instagram-%s.log' % u)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)


logger = logging.getLogger('InstagramBot')
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)


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
