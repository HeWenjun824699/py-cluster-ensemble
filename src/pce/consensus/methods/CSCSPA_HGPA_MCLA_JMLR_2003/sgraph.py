import os
import subprocess
import platform
import numpy as np

def sgraph(k, dataname):
    folder = os.path.dirname(os.path.abspath(__file__))
    
    # Determine executable
    if platform.system() == 'Windows':
        pmetis_cmd = os.path.join(folder, 'pmetis.exe')
        shmetis_cmd = os.path.join(folder, 'shmetis.exe')
    else:
        pmetis_cmd = os.path.join(folder, 'pmetis')
        shmetis_cmd = os.path.join(folder, 'shmetis')
    
    if dataname is None:
        dataname = os.path.join(folder, 'tmp', 'graph0')

    resultname = f"{dataname}.part.{k}"
    
    try:
        lastchar = int(dataname[-1])
    except ValueError:
        lastchar = 0
        
    if lastchar < 2:
        cmd = [pmetis_cmd, dataname, str(k)]
    else:
        ubfactor = 5
        cmd = [shmetis_cmd, dataname, str(k), str(ubfactor)]
        
    # Run command
    try:
        # Suppress output to keep it clean, or allow it for debugging
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"sgraph: partitioning failed. Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown'}")
        return None
    except FileNotFoundError:
         print(f"sgraph: executable not found: {cmd[0]}")
         return None

    # Read result
    labels = []
    if os.path.exists(resultname):
        with open(resultname, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(int(line))
        
        # Cleanup result file
        try:
            os.remove(resultname)
        except OSError:
            pass
    else:
        print(f"sgraph: result file {resultname} not found.")
        return None

    return np.array(labels)
