import sys
import os

# Ensure project root is importable as package during tests
ROOT = '/workspace'
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)