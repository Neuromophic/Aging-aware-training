import logging
import os

def GetMessageLogger(args, setup):
    
    if not os.path.exists(f'{args.logfilepath}/'):
        os.makedirs(f'{args.logfilepath}/')

    LOG_FORMAT = "%(asctime)s - %(levelname)s -\n%(message)s\n"
    logging.basicConfig(filename=f'{args.logfilepath}/{setup}.log', format=LOG_FORMAT)
    
    SETUP = vars(args)
    msg = ''
    for k, v in SETUP.items():
        msg = msg + str(k).ljust(20, ' ') + str(v) + '\n'
    
    msglogger = logging.getLogger()
    
    if args.loglevel == 'info':
        msglogger.setLevel(logging.INFO)
    elif args.loglevel == 'debug':
        msglogger.setLevel(logging.DEBUG)
        
    msglogger.info(msg)
    return msglogger

def PrintParser(args):
    args = vars(args)
    for k, v in args.items():
        print(str(k).ljust(20, ' ') + str(v))