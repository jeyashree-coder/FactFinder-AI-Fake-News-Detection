import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app.ml import model_manager
from app.database import init_db

init_db()
out1 = model_manager.predict("This is definitely real news about local events yesterday, verified by journalist.")
print("Test 1 (Real):", out1.label, out1.confidence, out1.details)

out2 = model_manager.predict("ALIENS LANDED ON THE MOON! HILLARY CLINTON SPOTTED WITH UFO! YOU WON'T BELIEVE THIS!")
print("Test 2 (Fake):", out2.label, out2.confidence, out2.details)
