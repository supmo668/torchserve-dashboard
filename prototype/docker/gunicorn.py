import multiprocessing
workers = max(4, multiprocessing.cpu_count()//4)
# Gunicorn config variables
bind = '0.0.0.0:8000'
loglevel = "info"
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 120
keepalive = 5
threads = 2