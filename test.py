import os
import subprocess


def main():
    print("Running tests!\n")

    directory_path = os.path.abspath("./data")
    output_dir = os.path.abspath("./output")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for entry in sorted(os.listdir(directory_path)):
        full_path = os.path.join(directory_path, entry)
        if not os.path.isfile(full_path):
            continue

        stem, _ = os.path.splitext(entry)
        annotated = os.path.join(output_dir, f"{stem}_annotated.mp4")
        summary = os.path.join(output_dir, f"{stem}.json")

        command = [
            "python3", "gait_detect.py",
            "--input", full_path,
            "--output", annotated,
            "--output-json", summary,
        ]

        print(f"\nProcessing file: {entry}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            print(result.stdout)
            if result.returncode != 0 and result.stderr:
                print(result.stderr)
            print(f"Return code: {result.returncode}")
        except FileNotFoundError:
            print(f"Error: command '{command[0]}' not found.")
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            continue

        results.append((full_path, result.returncode))

    return results


if __name__ == "__main__":
    main()
