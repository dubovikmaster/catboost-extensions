import requests
import logging
import _thread as thread
import threading
from functools import wraps

import platform
import signal
import os

from catboost import Pool

logger = logging.getLogger('utils')


def send_msgs_to_telegram(msgs: str, api_token: str, chat_id: int):
    url = f"https://api.telegram.org/bot{api_token}/sendMessage?chat_id={chat_id}&text={msgs}"
    try:
        response = requests.post(url)
    except Exception as e:
        logger.warning(f'Error while sending message to telegram: {e}')


def make_scorer(model, x, y, score=None, cat_features=None, text_features=None):
    iterations = model.get_param('iterations')
    if iterations is None:
        iterations = 1000
    if score is None:
        score = model.get_param('loss_function')
    return model.eval_metrics(
        Pool(x, y, cat_features=cat_features, text_features=text_features),
        score, ntree_start=iterations - 1)[score][-1]


def stop_function():
    if platform.system() == 'Windows':
        thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def stopit_after_timeout(s, raise_exception=True, exception=TimeoutError):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = threading.Timer(s, stop_function)
            try:
                timer.start()
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                msg = f'function \"{func.__name__}\" took longer than {s} s.'
                if raise_exception:
                    raise exception(msg)
                result = msg
            finally:
                timer.cancel()
            return result

        return wrapper

    return actual_decorator
