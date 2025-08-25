import numpy as np
import pyopencl as cl

from genepro.node import Node

class OpenCLTreeRunner:
    """
    Class to run a tree on an OpenCL device.

    Parameters
    ----------
    tree : Node
        The root node of the tree to be evaluated.
    X : np.ndarray
        Input data to evaluate the tree on.
    cl_ctx : cl.Context
        OpenCL context for the device.
    cl_queue : cl.CommandQueue
        Command queue for executing OpenCL commands.
    """

    def __init__(self, X: np.ndarray):
        self.cl_ctx = None
        for platform in cl.get_platforms():
            if platform.name == "Clover":
                continue  # Skip Clover platform as it does not support all features
            self.cl_ctx = cl.Context([platform.get_devices()[0]])
            break  # Break after the first suitable platform/device

        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        # Prepare input buffer
        X_f32 = X.astype(np.float32)
        self.X = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_f32)

        # Prepare output shape
        self.sample_count = X.shape[0]
        self.feature_count = X.shape[1]


    def run(self, tree: Node) -> np.ndarray:
        """
        Run a tree on the OpenCL device.

        Parameters
        ----------
        tree : Node
            The root node of the tree to be evaluated.
        X : np.ndarray
            Input data to evaluate the tree on.
        cl_ctx : cl.Context
            OpenCL context for the device.
        cl_queue : cl.CommandQueue
            Command queue for executing OpenCL commands.

        Returns
        -------
        np.ndarray
            The output of the tree evaluation.
        """

        # Generate OpenCL code for the tree evaluation
        cl_code = f"""
        __kernel void evaluate_tree(__global const float* X, __global float* output)
        {{
            int idx = get_global_id(0);
            int feature_count = {self.feature_count};
            float result = 0.0f;
            result = { tree.to_opencl() };
            output[idx] = result;
        }}
        """
        
        # Create OpenCL program
        program = cl.Program(self.cl_ctx, cl_code).build()

        # Prepare output buffer
        output_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, np.dtype(np.float32).itemsize * self.sample_count)

        # Execute the OpenCL kernel
        program.evaluate_tree(self.cl_queue, (self.sample_count,), None, self.X, output_buf)

        # Read the output back to host
        output = np.empty(self.sample_count, dtype=np.float32)
        cl.enqueue_copy(self.cl_queue, output, output_buf)

        return output