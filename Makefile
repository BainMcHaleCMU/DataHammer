.PHONY: backend backend-install backend-stop backend-logs

# Install backend dependencies
backend-install:
        cd backend && pip install -r requirements.txt

# Start the backend service
backend: backend-install
        cd backend && python main.py

# Start the backend service in the background
backend-daemon: backend-install
        cd backend && nohup python main.py > backend.log 2>&1 &
        @echo "Backend started in background. Check backend/backend.log for logs."

# Stop the backend service
backend-stop:
        @echo "To stop the backend, find the process ID with 'ps aux | grep \"python main.py\"' and kill it with 'kill <PID>'."

# View backend logs
backend-logs:
        @if [ -f backend/backend.log ]; then \
                tail -f backend/backend.log; \
        else \
                echo "Log file not found. Start the backend with 'make backend-daemon' first."; \
        fi

# Clean up
clean:
        @echo "Cleaning up..."
        @if [ -f backend/backend.log ]; then rm backend/backend.log; fi