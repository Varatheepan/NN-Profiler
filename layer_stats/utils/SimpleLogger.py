import logging

def SimpleLogger( args,logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create file handler which logs info messages
    fh = logging.FileHandler(logfile, 'w', 'utf-8')
    if args.log_level == "DEBUG":
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    formatter = logging.Formatter('- %(name)s - %(levelname)-8s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if args.log_level == "DEBUG":
        # create console handler with a debug log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger