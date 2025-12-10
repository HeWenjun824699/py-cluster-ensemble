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
        # Removed check=True to handle non-zero exit codes gracefully if file is produced
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        print(f"sgraph: partitioning failed (result file missing). Command: {' '.join(cmd)}")
        if result.returncode != 0:
            print(f"Exit Code: {result.returncode}")
            print(f"Error (stderr): {result.stderr.decode('utf-8') if result.stderr else 'None'}")
            print(f"Output (stdout): {result.stdout.decode('utf-8') if result.stdout else 'None'}")
        return None

    return np.array(labels)
