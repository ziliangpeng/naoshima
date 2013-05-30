from datetime import datetime

from decl import detect_fuzzy

def main():
    start_time = datetime.now()

    ret = detect_fuzzy('.', fuzzy=True, verbose=True)
    print ''
    for item in ret:
        print item

    end_time = datetime.now()
    print 'time spent:', end_time - start_time


if __name__ == '__main__':
    main()
