.PHONY: backend

backend:
        cd backend && pip install -r requirements.txt && python main.py
