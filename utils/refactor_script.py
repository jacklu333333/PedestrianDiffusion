import os
import re
import shutil

# --- CONFIGURATION ---
SOURCE_FILE = "utils/mcallbacks.py"  # Ensure your source file is named this
PACKAGE_NAME = "mcallbacks"  # The directory/package name
COMMON_FILENAME = "common_imports.py"
UTILS_FILENAME = "utils.py"


def ensure_dirs():
    if os.path.exists(PACKAGE_NAME):
        shutil.rmtree(PACKAGE_NAME)
    os.makedirs(PACKAGE_NAME)


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    imports = []
    functions = []
    classes = []

    current_block = []
    block_type = None  # 'import', 'function', 'class'
    block_name = None
    parent_class = None

    # Regex patterns to detect the START of a block
    # We ignore indentation in the regex, but check it in logic
    class_pattern = re.compile(r"^class\s+(\w+)(?:\(([^)]+)\))?:")
    func_pattern = re.compile(r"^def\s+(\w+)\s*\(")
    import_pattern = re.compile(r"^(from\s+|import\s+)")
    decorator_pattern = re.compile(r"^@")  # Handle decorators if needed

    for line in lines:
        stripped = line.strip()
        # Calculate indentation: essentially 0 if matching start of line
        indent_level = len(line) - len(line.lstrip())

        # Check if this line is the START of a new block
        # It must be at indent 0 and match one of our patterns
        class_match = class_pattern.match(line) if indent_level == 0 else None
        func_match = func_pattern.match(line) if indent_level == 0 else None
        import_match = import_pattern.match(line) if indent_level == 0 else None

        is_new_start = class_match or func_match or import_match

        if is_new_start:
            # 1. FLUSH the previous block before starting the new one
            if current_block:
                if block_type == "import":
                    imports.extend(current_block)
                elif block_type == "function":
                    functions.append({"name": block_name, "code": current_block})
                elif block_type == "class":
                    classes.append(
                        {
                            "name": block_name,
                            "parent": parent_class,
                            "code": current_block,
                        }
                    )

                # Reset
                current_block = []
                block_type = None
                block_name = None
                parent_class = None

            # 2. START the new block
            if class_match:
                block_type = "class"
                block_name = class_match.group(1)
                parents = class_match.group(2)
                parent_class = parents.split(",")[0].strip() if parents else None
            elif func_match:
                block_type = "function"
                block_name = func_match.group(1)
            elif import_match:
                block_type = "import"

            current_block.append(line)

        else:
            # 3. CONTINUATION
            # If it's not a new start, it belongs to the previous block.
            # This captures indented lines, comments, AND closing parentheses ')' at indent 0.
            if current_block:
                current_block.append(line)
            else:
                # If we haven't started any block yet (e.g. file header comments),
                # treat them as imports/global scope.
                imports.append(line)

    # 4. FLUSH FINAL BLOCK
    if current_block:
        if block_type == "import":
            imports.extend(current_block)
        elif block_type == "function":
            functions.append({"name": block_name, "code": current_block})
        elif block_type == "class":
            classes.append(
                {"name": block_name, "parent": parent_class, "code": current_block}
            )

    return imports, functions, classes


def clean_imports(import_lines):
    """
    Attempt to fix relative imports in the extracted lines.
    Replaces 'from .' with 'from model.'
    """
    cleaned = []
    for line in import_lines:
        new_line = re.sub(r"^from \.", f"from {PACKAGE_NAME}.", line)
        cleaned.append(new_line)
    return cleaned


def write_common_file(imports):
    path = os.path.join(PACKAGE_NAME, COMMON_FILENAME)
    cleaned_imports = clean_imports(imports)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_imports)


def write_utils_file(functions):
    if not functions:
        return
    path = os.path.join(PACKAGE_NAME, UTILS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"from {PACKAGE_NAME}.{COMMON_FILENAME[:-3]} import *\n\n")
        for func in functions:
            f.writelines(func["code"])
            f.write("\n\n")


def write_class_files(classes):
    internal_classes = {c["name"] for c in classes}

    for cls in classes:
        filename = f"{cls['name']}.py"
        filepath = os.path.join(PACKAGE_NAME, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            # Absolute Imports
            f.write(f"from {PACKAGE_NAME}.{COMMON_FILENAME[:-3]} import *\n")
            f.write(f"from {PACKAGE_NAME}.{UTILS_FILENAME[:-3]} import *\n")

            # Inherited Class Import
            if cls["parent"]:
                simple_parent = cls["parent"].split(".")[-1]
                if simple_parent in internal_classes:
                    f.write(
                        f"from {PACKAGE_NAME}.{simple_parent} import {simple_parent}\n"
                    )

            # Specific Hardcoded Dependencies (Dependencies used inside methods)
            if "cycleGan" in cls["name"]:
                if "UNet1D" in internal_classes:
                    f.write(f"from {PACKAGE_NAME}.UNet1D import UNet1D\n")
                if "Discriminator1D" in internal_classes:
                    f.write(
                        f"from {PACKAGE_NAME}.Discriminator1D import Discriminator1D\n"
                    )

            if (
                "VAEGeneratorModule" in cls["name"]
                and "timeAEModel" in internal_classes
            ):
                f.write(f"from {PACKAGE_NAME}.timeAEModel import timeAEModel\n")

            f.write("\n")
            f.writelines(cls["code"])


def create_init_file(classes):
    path = os.path.join(PACKAGE_NAME, "__init__.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {PACKAGE_NAME} package\n\n")

        # Expose classes so user can do `from model import ClassName`
        for cls in classes:
            f.write(f"from {PACKAGE_NAME}.{cls['name']} import {cls['name']}\n")

        f.write(f"\nfrom {PACKAGE_NAME}.{UTILS_FILENAME[:-3]} import *\n")


def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: '{SOURCE_FILE}' not found.")
        return

    print(f"Parsing {SOURCE_FILE}...")
    ensure_dirs()

    imports, functions, classes = parse_file(SOURCE_FILE)

    print(f"  - Detected {len(imports)} import lines.")
    print(f"  - Detected {len(functions)} global functions.")
    print(f"  - Detected {len(classes)} classes.")

    print("Writing files...")
    write_common_file(imports)
    write_utils_file(functions)
    write_class_files(classes)
    create_init_file(classes)

    print("-" * 30)
    print(f"Refactoring complete. Folder '{PACKAGE_NAME}' created.")
    print(f"Usage example:\n  from {PACKAGE_NAME} import {classes[-1]['name']}")


if __name__ == "__main__":
    main()
