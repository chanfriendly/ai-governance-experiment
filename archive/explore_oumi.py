# explore_oumi.py

import oumi
import inspect

print("Exploring Oumi library structure...")
print(f"Oumi version: {getattr(oumi, '__version__', 'Unknown')}")
print("\nOumi directory:", oumi.__file__)

print("\nOumi attributes:")
for attr in dir(oumi):
    if not attr.startswith('__'):  # Skip internal attributes
        print(f"- {attr}")
        
        # If this is a function or class, show its signature
        attr_value = getattr(oumi, attr)
        if inspect.isfunction(attr_value) or inspect.isclass(attr_value):
            try:
                print(f"  Signature: {inspect.signature(attr_value)}")
            except ValueError:
                print(f"  Signature: Could not determine")