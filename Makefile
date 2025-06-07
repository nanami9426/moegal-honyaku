.PHONY: all
all: install run

.PHONY: run
run:
	@echo "Running the application..."
	uv run main.py


.PHONY: install
install:
	@echo "Installing dependencies..."
	uv sync

