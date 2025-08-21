"""Test matplotlib step by step"""

print("1. Testing basic imports...")
import os
import sys
import numpy as np
print("✅ Basic imports successful")

print("2. Testing matplotlib import...")
import matplotlib
print(f"✅ Matplotlib version: {matplotlib.__version__}")

print("3. Setting backend...")
matplotlib.use('Agg')
print("✅ Backend set to Agg")

print("4. Testing pyplot...")
import matplotlib.pyplot as plt
print("✅ Pyplot imported")

print("5. Testing seaborn...")
import seaborn as sns
print("✅ Seaborn imported")

print("6. Testing pandas...")
import pandas as pd
print("✅ Pandas imported")

print("7. Creating simple plot...")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Test Plot")
    
    # Save to file
    os.makedirs("tests/test_results/graphs", exist_ok=True)
    plt.savefig("tests/test_results/graphs/test_plot.png", dpi=300)
    plt.close()
    print("✅ Test plot created and saved")
    
except Exception as e:
    print(f"❌ Plot creation failed: {e}")
    import traceback
    traceback.print_exc()

print("8. All tests completed! 🎉")
