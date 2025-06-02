import os
from apscheduler.schedulers.background import BackgroundScheduler
from .logger import logger
from .anomaly_handler import AnomalyHandler
from .recommendation_handler import RecommendationHandler

class Monitor:
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.anomaly_handler = AnomalyHandler(self.base_dir)
        self.recommendation_handler = RecommendationHandler(self.base_dir)
        self.scheduler = BackgroundScheduler()

    def start_monitor(self):
        # Job 1: Check anomaly logs
        self.scheduler.add_job(
            self.anomaly_handler.check_logs,
            'interval', # Repeatedly at a fixed interval 
            minutes=0.1,
            id='anomaly_monitor_job',
            max_instances=1,
            replace_existing=True
        )

        # Job 2: Check recommendation logs
        self.scheduler.add_job(
            self.recommendation_handler.check_logs,
            'interval', # Repeatedly at a fixed interval
            minutes=0.1,
            id='recommendation_monitor_job',
            max_instances=1,
            replace_existing=True
        )

        self.scheduler.start()
        logger.info("Unified monitor scheduler started.")

