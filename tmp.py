import csv

def normalize_time(input_file, output_file):
    """
    Normaliza os valores de tempo para começar em 0.
    
    Args:
        input_file: caminho do arquivo de entrada
        output_file: caminho do arquivo de saída
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = [line for line in lines[1:] if line.strip() and not line.strip() == ';;']
    
    times = []
    rows = []
    for line in data_lines:
        parts = line.strip().rstrip(';;').split(',')
        if len(parts) > 0:
            times.append(float(parts[0]))
            rows.append(parts)
    
    if times:
        first_time = times[0]
        
        # Escreve o arquivo de saída
        with open(output_file, 'w') as f:
            f.write(header)
            for i, row in enumerate(rows):
                normalized_time = times[i] - first_time
                row[0] = f"{normalized_time:.3f}"
                f.write(','.join(row) + ';;\n')
    
    print(f"Tempo normalizado! Primeiro valor era {first_time:.3f}, agora é 0.000")

if __name__ == "__main__":
    input_file = "tracked_data.txt" 
    output_file = "tracked_data_normalizados.csv"
    
    normalize_time(input_file, output_file)
