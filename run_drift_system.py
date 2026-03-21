import os

# Monitoring
os.system("python monitoring/drift_report.py")
os.system("python monitoring/performance_monitor.py")

# Trigger retraining
os.system("python retraining/trigger_engine.py")
