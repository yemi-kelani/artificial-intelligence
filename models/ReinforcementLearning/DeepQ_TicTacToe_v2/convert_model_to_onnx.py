"""
ONNX Model Conversion Script for DeepQ TicTacToe Agent v2

Converts a trained PyTorch model (.pt) to ONNX format for deployment
in browsers, mobile apps, or backend servers.

Usage:
    python convert_model_to_onnx.py <model_path> [--output <output_path>]

Examples:
    python convert_model_to_onnx.py trained_models/ReinforcementLearning/TicTacToeV2/FINAL_MODEL.pt
    python convert_model_to_onnx.py FINAL_MODEL.pt --output custom_output.onnx
"""

import os
import sys
import argparse

import onnx
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


class DeepQAgentOnnxVersion(nn.Module):
    """
    Simplified DeepQAgent for ONNX export.

    This class mirrors the policy_network architecture from DeepQAgent v2
    but without training-related components (memory, target network, etc.).
    """

    def __init__(
        self,
        state_space: int = 9,
        action_space: int = 9,
        hidden_size: int = 256,
        dropout: float = 0.15,
    ):
        super(DeepQAgentOnnxVersion, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        # Match the exact architecture from DeepQAgent.create_network()
        self.model = nn.Sequential(
            nn.Linear(state_space, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_space)
        )

    def forward(self, state):
        return self.model(state)


# V2 Hyperparameters (from train.ipynb)
STATE_SPACE = 9
ACTION_SPACE = 9
HIDDEN_SIZE = 256
DROPOUT = 0.15

# Default paths
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models", "ReinforcementLearning", "TicTacToeV2")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "trained_models", "onnx_models", "TicTacToeAgentV2.onnx")


def convert_pt_to_onnx(
    model_path: str,
    output_path: str,
    input_size: tuple = (1, 9),
    verbose: bool = True
):
    """
    Convert a PyTorch .pt model to ONNX format.

    Args:
        model_path: Path to the .pt model file
        output_path: Path for the output .onnx file
        input_size: Input tensor shape (batch_size, state_space)
        verbose: Whether to print detailed output

    Returns:
        bool: True if conversion successful, False otherwise
    """
    device = torch.device("cpu")  # ONNX export works best on CPU

    # Create model with v2 architecture
    model = DeepQAgentOnnxVersion(
        state_space=STATE_SPACE,
        action_space=ACTION_SPACE,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT
    )

    # Load trained weights
    # The saved model contains policy_network state dict directly
    if verbose:
        print(f"Loading model from: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Map policy_network weights to our simplified model
    # DeepQAgent saves policy_network.state_dict() which has keys like:
    # '0.weight', '0.bias', '3.weight', '3.bias', '6.weight', '6.bias'
    model.model.load_state_dict(state_dict)
    model.eval()

    if verbose:
        print(f"Model loaded successfully")
        print(f"Architecture: {STATE_SPACE} -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> {ACTION_SPACE}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Create dummy input for tracing
    dummy_input = torch.zeros(input_size, dtype=torch.float32)

    # Export to ONNX
    if verbose:
        print(f"\nExporting to ONNX: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=verbose,
        input_names=["board_state"],
        output_names=["q_values"],
        dynamic_axes={
            "board_state": {0: "batch_size"},
            "q_values": {0: "batch_size"}
        },
        opset_version=11
    )

    # Validate the exported model
    if verbose:
        print("\nValidating ONNX model...")

    onnx_model = onnx.load(output_path)

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f"ONNX validation failed: {e}")
        return False

    if verbose:
        print("ONNX model is valid!")
        print("\nModel Graph:")
        print(onnx.helper.printable_graph(onnx_model.graph))

    # Run inference test
    if verbose:
        print("\n" + "="*50)
        print("Running inference test...")

    ort_session = ort.InferenceSession(output_path)

    # Test with empty board
    test_inputs = [
        ("Empty board", np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)),
        ("X in center", np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0]], dtype=np.float32)),
        ("X corner, O center", np.array([[-1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)),
    ]

    for name, board_state in test_inputs:
        outputs = ort_session.run(None, {"board_state": board_state})
        q_values = outputs[0][0]
        best_action = np.argmax(q_values)

        if verbose:
            print(f"\n{name}:")
            print(f"  Input:  {board_state[0]}")
            print(f"  Q-values: {np.round(q_values, 4)}")
            print(f"  Best action: {best_action} (row={best_action//3}, col={best_action%3})")

    if verbose:
        print("\n" + "="*50)
        print(f"Conversion complete!")
        print(f"Output saved to: {output_path}")
        file_size = os.path.getsize(output_path) / 1024
        print(f"File size: {file_size:.2f} KB")

    return True


def resolve_model_path(model_path: str) -> str:
    """
    Resolve the model path, checking multiple locations.

    Args:
        model_path: User-provided path (can be filename or full path)

    Returns:
        Resolved absolute path to the model file
    """
    # If it's already an absolute path and exists, use it
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path

    # If it exists as a relative path from current directory
    if os.path.exists(model_path):
        return os.path.abspath(model_path)

    # Try in the default model directory
    default_path = os.path.join(DEFAULT_MODEL_DIR, model_path)
    if os.path.exists(default_path):
        return default_path

    # Try adding .pt extension
    if not model_path.endswith(".pt"):
        with_ext = model_path + ".pt"
        if os.path.exists(with_ext):
            return os.path.abspath(with_ext)
        default_with_ext = os.path.join(DEFAULT_MODEL_DIR, with_ext)
        if os.path.exists(default_with_ext):
            return default_with_ext

    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        f"Searched locations:\n"
        f"  - {os.path.abspath(model_path)}\n"
        f"  - {os.path.join(DEFAULT_MODEL_DIR, model_path)}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="convert_model_to_onnx",
        description="Convert a trained DeepQ TicTacToe v2 model to ONNX format.",
        epilog="Example: python convert_model_to_onnx.py FINAL_MODEL.pt"
    )

    parser.add_argument(
        "model_path",
        help="Path to the .pt model file (can be filename or full path)"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for ONNX file (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    try:
        model_path = resolve_model_path(args.model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    success = convert_pt_to_onnx(
        model_path=model_path,
        output_path=args.output,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
