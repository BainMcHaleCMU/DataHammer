.PHONY: backend

backend:
        cd backend && \
        python -m venv venv && \
        . venv/bin/activate && \
        pip install -r requirements.txt && \
        python main.py
