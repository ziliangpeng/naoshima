import traceback
import sys

def err():
    try:
        # do something wrong
        a = int('OK')
    except BaseException as e:
        try:
            print("Error happened")
            # traceback.print_tb(e, file=sys.stdout)
            print(traceback.format_exc())
        except BaseException:
            print("Error in tracing back")

err()
