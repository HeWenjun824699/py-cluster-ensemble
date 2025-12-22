import numpy as np
from twist.wshrinkObj import wshrinkObj

def test_wshrinkObj():
    print("Testing wshrinkObj...")
    
    # Mock data dimensions
    n = 10
    sX = [n, n, 2]
    
    # Create random data
    np.random.seed(42)
    t = np.random.rand(n * n * 2)
    
    rho = 0.1
    isWeight = 0
    mode = 3
    
    # Run wshrinkObj
    try:
        x_out, objV = wshrinkObj(t, rho, sX, isWeight, mode)
        
        print("Execution successful.")
        print(f"Objective Value: {objV}")
        print(f"Output shape: {x_out.shape}")
        print(f"Output type: {x_out.dtype}")
        
        # Check if output is real (it might be complex with 0j due to IFFT inside, 
        # but TensorEnsemble handles the cast. Let's see what wshrinkObj returns raw)
        if np.iscomplexobj(x_out):
            print("Output is complex (expected raw output from IFFT).")
            print(f"Max imaginary part: {np.max(np.abs(x_out.imag))}")
        else:
            print("Output is real.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wshrinkObj()
