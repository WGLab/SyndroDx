"""Persistent supervisor entry point; no queue-wait or training-segment expiry."""
from lease_supervisor import main

if __name__ == "__main__":
    main()
