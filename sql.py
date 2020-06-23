"""
Writing features to SQL server
"""
import argparse

from seizurecast.postgresql import run_sql_task

VERSION = 'v1.0.0'


def parse_args():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("task", help="task")
    parser.add_argument("start", help="starting index")
    parser.add_argument("end", help="ending index")
    parser.add_argument("-V", "--version", help="show program version",
                        action="version", version=VERSION)
    parser.add_argument("-v", "--verbose",
                        help="enable verbose mode",
                        action="store_true")
    # parser.set_defaults(verbose=False,
    #                     config='./config.ini',
    #                     infile='./data/svdemo-3bkg-3pre-3bkg-3pre.txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_sql_task(indexes=(int(args.start), int(args.end)), task=args.task)
