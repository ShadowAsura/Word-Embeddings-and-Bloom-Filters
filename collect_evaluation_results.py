import os
import subprocess
import re
import json

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(stderr.decode())
    return stdout.decode()

def parse_analogy_output(output):
    lines = output.split('\n')
    analogy_results = []
    for line in lines:
        match = re.match(r'\s*(\d+)\. (\w+)\s+\(similarity=([0-9.]+)\)', line)
        if match:
            analogy_results.append({'rank': int(match.group(1)), 'word': match.group(2), 'similarity': float(match.group(3))})
    return analogy_results

def parse_evaluate_output(output):
    accuracy_match = re.search(r'Total accuracy: ([0-9.]+)\%\s+\(([0-9]+)/([0-9]+)\)', output)
    if accuracy_match:
        total_accuracy = float(accuracy_match.group(1))
        correct = int(accuracy_match.group(2))
        total_questions = int(accuracy_match.group(3))
        return {'total_accuracy': total_accuracy, 'correct': correct, 'total_questions': total_questions}
    return None

if __name__ == "__main__":
    iterations_to_evaluate = [10, 25, 50, 75, 100, 150]
    results = {}
    base_path = "c:\\Users\\Logan\\Scripts\\Random\\Bloom Filter\\Word-Embeddings-and-Bloom-Filters"

    for iter_num in iterations_to_evaluate:
        print(f"\n--- Evaluating Iteration {iter_num} ---")
        # Run word_analogy.py
        analogy_command = [
            "python", os.path.join(base_path, "eval", "word_analogy.py"),
            "--iteration", str(iter_num),
            "--vector_file", os.path.join(base_path, "data", "iterative_vectors", f'window_6_iter_{iter_num}_v3_32bit.json')
        ]
        analogy_output = run_command(analogy_command)
        parsed_analogy = parse_analogy_output(analogy_output)
        
        # Run evaluate.py
        evaluate_command = [
            "python", os.path.join(base_path, "eval", "evaluate.py"),
            "--iteration", str(iter_num),
            "--vector_file", os.path.join(base_path, "data", "iterative_vectors", f'window_6_iter_{iter_num}_v3_32bit.json')
        ]
        evaluate_output = run_command(evaluate_command)
        parsed_evaluate = parse_evaluate_output(evaluate_output)

        results[iter_num] = {
            'analogy': parsed_analogy,
            'evaluate': parsed_evaluate
        }

        # Print immediate results for feedback
        print(f"Analogy (king man queen): {parsed_analogy[0]['word']} (similarity={parsed_analogy[0]['similarity']:.4f})")
        if parsed_evaluate:
            print(f"Total Accuracy: {parsed_evaluate['total_accuracy']:.2f}%")
        else:
            print("Total Accuracy: N/A")

    print("\n--- Summary of Results ---")
    print("Iteration | Top-1 Analogy (king man queen) | Total Accuracy")
    print("-------------------------------------------------------------------")
    for iter_num, res in results.items():
        analogy_word = res['analogy'][0]['word'] if res['analogy'] else "N/A"
        analogy_sim = res['analogy'][0]['similarity'] if res['analogy'] else "N/A"
        accuracy = f"{res['evaluate']['total_accuracy']:.2f}%" if res['evaluate'] else "N/A"
        print(f"{iter_num:<9} | {analogy_word:<20} (sim={analogy_sim:.4f}) | {accuracy}")

    # Further analysis will be done after confirming the files are generated and this script runs.
