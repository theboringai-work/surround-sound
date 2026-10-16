#!/usr/bin/env python3
"""Build script to generate index.html (the model showcase page) from paper assets and class metadata."""
import os

def build():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print("Compiling index.html from templates/index_template.html...")
    with open("templates/index_template.html") as f_in, open("assets/classes_data.json") as f_data:
        template = f_in.read()
        data = f_data.read()

    rendered = template.replace("__CLASSES_JSON__", data)
    with open("index.html", "w") as f_out:
        f_out.write(rendered)

    print(f"index.html successfully generated ({len(rendered):,} bytes)!")

if __name__ == "__main__":
    build()
