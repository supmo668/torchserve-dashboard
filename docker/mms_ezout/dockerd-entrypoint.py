import os
import shlex
import subprocess
import sys
from subprocess import CalledProcessError

# from retrying import retry
from sagemaker_inference import model_server

handler_file = os.environ.get(
    'MODEL_HANDLER', '/opt/ml/code/model_handler.py') 
config_file = os.environ.get(
    'MMS_CONFIG_FILE', './config.properties') 
def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


# @retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_mms(handler_file=handler_file):
    # by default the number of workers per model is 1, but we can configure it through the
    # environment variable below if desired.
    # os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = '2'
    model_server.start_model_server(
        handler_service=f"{handler_file}:handle",
        # config_file=config_file
    )


def main(args=None):
    if sys.argv[1] == "serve":
        if args is None:
            _start_mms()
        else:
            print(f"Starting mms with args:{args}")
            _start_mms(args.handler_file)
    else:
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

    # prevent docker exit
    subprocess.call(["tail", "-f", "/dev/null"])

main()

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Start the API service.')
    parser.add_argument(
        '--loglevel', type=str, default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Logging level (default: info)')
    parser.add_argument(
        '--handler_file', type=str, default=Path(__file__).resolve().parent/'model_handler.py')
                        
    args = parser.parse_args()
    main(args)